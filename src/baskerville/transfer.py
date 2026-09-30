import json
import re
import h5py

import tensorflow as tf

from baskerville import layers
from baskerville import adapters


def param_count(layer, type="all"):
    if type not in ["all", "trainable", "non_trainable"]:
        raise ValueError("TYPE must be one of all, trainable, non_trainable")
    output = 0
    if type == "all":
        output = int(sum(tf.keras.backend.count_params(w) for w in layer.weights))
    elif type == "trainable":
        output = int(
            sum(tf.keras.backend.count_params(w) for w in layer.trainable_weights)
        )
    else:
        output = int(
            sum(tf.keras.backend.count_params(w) for w in layer.non_trainable_weights)
        )
    return output


def param_summary(model):
    trainable = param_count(model, type="trainable")
    non_trainable = param_count(model, type="non_trainable")
    print("total params:%d" % (trainable + non_trainable))
    print("trainable params:%d" % trainable)
    print("non-trainable params:%d" % non_trainable)


def keras2dict(model):
    layer_parent_dict = {}  # the parent layers of each layer in the old graph
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in layer_parent_dict:
                layer_parent_dict.update({layer_name: [layer.name]})
            else:
                if layer.name not in layer_parent_dict[layer_name]:
                    layer_parent_dict[layer_name].append(layer.name)
    return layer_parent_dict


# lora requires change model.h5 weight order.
# locon and ia3 don't modify model in place.
def var_reorder(weight_h5):
    # assumes weight_h5 model saved with seqnn_model.save()
    # [i.name for i in model.layers[30].weights] to check for multihead_attention layer weights order.
    # model.load_weights() load weights sequencially, assuming h5 weights are in the right order.
    # When inserting lora, multihead_attention layer weights order changed.
    # multihead_attention layer weights order is saved inside f['model_weights']['multihead_attention'].attrs
    # After saving the weight_merged model, we need to go into the weights.h5, and change the attrs in multihead attention.
    var_init_order = [
        "r_w_bias:0:0",
        "r_r_bias:0:0",
        "q_layer/kernel:0",
        "k_layer/kernel:0",
        "v_layer/kernel:0",
        "embedding_layer/kernel:0",
        "embedding_layer/bias:0",
        "r_k_layer/kernel:0",
    ]

    f = h5py.File(weight_h5, "r+")
    layers = [i for i in list(f["model_weights"].keys()) if "multihead_attention" in i]
    for l_name in layers:
        new_name_order = [l_name + "/" + i for i in var_init_order]
        f["model_weights"][l_name].attrs.modify(
            name="weight_names", value=new_name_order
        )
    f.close()


# houlsby requires architecture change.
# thus we need to modify json.
def modify_json(
    input_json, output_json, adapter, latent=8, se_rank=None, conv_select=None
):
    with open(input_json) as params_open:
        params = json.load(params_open)

    # houlsby
    if adapter == "adapterHoulsby":
        params["model"]["adapter"] = "houlsby"
        params["model"]["adapter_latent"] = latent

    # houlsby_se
    elif adapter == "houlsby_se":
        params["model"]["adapter"] = "houlsby_se"
        params["model"]["adapter_latent"] = latent
        params["model"]["se_rank"] = se_rank
        params["model"]["conv_select"] = conv_select

    else:
        raise ValueError("adapter must be adapterHoulsby or houlsby_se")

    ### output
    with open(output_json, "w") as params_open:
        json.dump(params, params_open, indent=4)


######################
# add houlsby layers #
######################
def add_houlsby(input_model, strand_pair, latent_size=8):
    # take seqnn_model as input
    # output a new seqnn_model object
    # only the adapter, and layer_norm are trainable

    ##################
    # houlsby layers #
    ##################
    houlsby_layers = []
    for i in range(len(input_model.layers) - 1):
        layer = input_model.layers[i]
        next_layer = input_model.layers[i + 1]
        if re.match("dropout", layer.name) and re.match("add", next_layer.name):
            houlsby_layers += [next_layer.name]

    ###################
    # construct model #
    ###################
    layer_parent_dict_old = keras2dict(input_model)
    # remove switch_reverse_layer
    to_fix = [i for i in layer_parent_dict_old if re.match("switch_reverse", i)]
    for i in to_fix:
        del layer_parent_dict_old[i]
    # create new graph
    layer_output_dict_new = {}  # the output tensor of each layer in the new graph
    layer_output_dict_new.update({input_model.layers[0].name: input_model.input})
    # Iterate over all layers after the input
    reverse_bool = None

    for layer in input_model.layers[1:-1]:
        # parent layers
        parent_layers = layer_parent_dict_old[layer.name]

        # layer inputs
        layer_input = [layer_output_dict_new[parent] for parent in parent_layers]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if re.match("stochastic_reverse_complement", layer.name):
            x, reverse_bool = layer(layer_input)

        # insert houlsby layer:
        elif layer.name in houlsby_layers:
            print("adapter added before:%s" % layer.name)
            x = adapters.AdapterHoulsby(latent_size=latent_size)(layer_input[1])
            x = layer([layer_input[0], x])

        else:
            x = layer(layer_input)

        # save the output tensor of every layer
        layer_output_dict_new.update({layer.name: x})

    final = layers.SwitchReverse(strand_pair)(
        [layer_output_dict_new[input_model.layers[-2].name], reverse_bool]
    )
    model_adapter = tf.keras.Model(inputs=input_model.inputs, outputs=final)

    # set trainable #
    for l in model_adapter.layers[:-2]:  # trunk
        if re.match("layer_normalization|adapter_houlsby", l.name):
            l.trainable = True
        else:
            l.trainable = False

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_adapter.layers:
        if l.name.startswith("adapter_houlsby"):
            params_added += param_count(l)
        elif l.name.startswith("layer_normalization"):
            params_added += param_count(l, type="trainable")
    print("params added/unfrozen by adapter_houlsby: %d" % params_added)

    return model_adapter


###############
# lora layers #
###############
def add_lora(input_model, rank=8, alpha=16, mode="default", report_param=True):
    # take seqnn.model as input
    # replace _q_layer, _v_layer in multihead_attention
    # optionally replace _k_layer, _embedding_layer
    if mode not in ["default", "full"]:
        raise ValueError("mode must be default or full")

    for layer in input_model.layers:
        if re.match("multihead_attention", layer.name):
            # default loRA
            layer._q_layer = adapters.Lora(
                layer._q_layer, rank=rank, alpha=alpha, trainable=True
            )
            layer._v_layer = adapters.Lora(
                layer._v_layer, rank=rank, alpha=alpha, trainable=True
            )
            # full loRA
            if mode == "full":
                layer._k_layer = adapters.Lora(
                    layer._k_layer, rank=rank, alpha=alpha, trainable=True
                )
                layer._embedding_layer = adapters.Lora(
                    layer._embedding_layer, rank=rank, alpha=alpha, trainable=True
                )

    input_model(input_model.input)  # initialize new variables

    # freeze all params but lora
    for layer in input_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())
        if len(lst_of_sublayers) == 1:
            if layer.name in ["lora_a", "lora_b"]:
                layer.trainable = True
            else:
                layer.trainable = False

    ### bias terms need to be frozen separately
    for layer in input_model.layers:
        if re.match("multihead_attention", layer.name):
            layer._r_w_bias = tf.Variable(
                layer._r_w_bias, trainable=False, name=layer._r_w_bias.name
            )
            layer._r_r_bias = tf.Variable(
                layer._r_r_bias, trainable=False, name=layer._r_r_bias.name
            )

    # set final head to be trainable
    input_model.layers[-2].trainable = True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in input_model.layers:
        if re.match("multihead_attention", l.name):
            params_added += param_count(l._q_layer.down_layer)
            params_added += param_count(l._q_layer.up_layer)
            params_added += param_count(l._v_layer.down_layer)
            params_added += param_count(l._v_layer.up_layer)
            if mode == "full":
                params_added += param_count(l._k_layer.down_layer)
                params_added += param_count(l._k_layer.up_layer)
                params_added += param_count(l._embedding_layer.down_layer)
                params_added += param_count(l._embedding_layer.up_layer)

    if report_param:
        print("params added/unfrozen by lora: %d" % params_added)


###############
# lora layers #
###############
def add_lora_conv(input_model, conv_select=None):
    # add lora layers
    add_lora(input_model, rank=8, alpha=16, mode="default", report_param=False)

    # list all conv layers
    conv_layers = []
    for layer in input_model.layers:
        if re.match("conv1d", layer.name):
            conv_layers += [layer.name]
    if conv_select is None:
        conv_select = len(conv_layers)
    if conv_select > len(conv_layers):
        raise ValueError(
            "conv_select must be less than number of conv layers %d." % len(conv_layers)
        )

    # set conv layers trainable
    trainable_conv = conv_layers[-conv_select:]
    for layer in input_model.layers:
        if layer.name in trainable_conv:
            layer.trainable = True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in input_model.layers:
        if re.match("multihead_attention", l.name):
            params_added += param_count(l._q_layer.down_layer)
            params_added += param_count(l._q_layer.up_layer)
            params_added += param_count(l._v_layer.down_layer)
            params_added += param_count(l._v_layer.up_layer)
        elif l.name in trainable_conv:
            params_added += param_count(l)

    print("params added/unfrozen by lora_conv: %d" % params_added)


# merge lora weights #
def merge_lora_layer(lora_layer):
    down_weights = lora_layer.down_layer.kernel
    up_weights = lora_layer.up_layer.kernel
    increment_weights = (
        tf.einsum("ab,bc->ac", down_weights, up_weights) * lora_layer.scale
    )
    lora_layer.original_layer.kernel.assign_add(increment_weights)
    return lora_layer.original_layer


def merge_lora(input_model):
    for layer in input_model.layers:
        if "multihead_attention" in layer.name:
            if isinstance(layer._q_layer, adapters.Lora):
                layer._q_layer = merge_lora_layer(layer._q_layer)
            if isinstance(layer._v_layer, adapters.Lora):
                layer._v_layer = merge_lora_layer(layer._v_layer)
            if isinstance(layer._k_layer, adapters.Lora):
                layer._k_layer = merge_lora_layer(layer._k_layer)
            if isinstance(layer._embedding_layer, adapters.Lora):
                layer._embedding_layer = merge_lora_layer(layer._embedding_layer)
    input_model(input_model.input)


##############
# IA3 layers #
##############
def add_ia3(input_model, strand_pair):
    # add to kv layers #
    for layer in input_model.layers:
        if re.match("multihead_attention", layer.name):
            layer._k_layer = adapters.IA3(layer._k_layer, trainable=True)
            layer._v_layer = adapters.IA3(layer._v_layer, trainable=True)

    # add to ff layer #
    # save old graph to dictionary
    layer_parent_dict_old = keras2dict(input_model)

    # remove switch_reverse_layer
    to_fix = [i for i in layer_parent_dict_old if re.match("switch_reverse", i)]
    for i in to_fix:
        del layer_parent_dict_old[i]

    # create new graph
    layer_output_dict_new = {}  # the output tensor of each layer in the new graph
    layer_output_dict_new.update({input_model.layers[0].name: input_model.input})

    # Iterate over all layers after the input
    reverse_bool = None
    for layer in input_model.layers[1:-1]:
        # get layer inputs
        parent_layers = layer_parent_dict_old[layer.name]
        layer_input = [layer_output_dict_new[parent] for parent in parent_layers]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # construct
        if re.match("stochastic_reverse_complement", layer.name):
            x, reverse_bool = layer(layer_input)
        # transformer ff down-project layer (1536 -> 768):
        elif re.match("dense", layer.name) and layer.input_shape[-1] == 1536:
            x = adapters.IA3_ff(layer, trainable=True)(layer_input)
        else:
            x = layer(layer_input)

        # save layers to dictionary
        layer_output_dict_new.update({layer.name: x})

    final = layers.SwitchReverse(strand_pair)(
        [layer_output_dict_new[input_model.layers[-2].name], reverse_bool]
    )
    model_adapter = tf.keras.Model(inputs=input_model.inputs, outputs=final)

    # set trainable #
    for layer in model_adapter._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())
        if len(lst_of_sublayers) == 1:
            if layer.name in ["ia3", "ia3_ff"]:
                layer.trainable = True
            else:
                layer.trainable = False

    ### bias terms need to be frozen separately
    for layer in model_adapter.layers:
        if re.match("multihead_attention", layer.name):
            layer._r_w_bias = tf.Variable(
                layer._r_w_bias, trainable=False, name=layer._r_w_bias.name
            )
            layer._r_r_bias = tf.Variable(
                layer._r_r_bias, trainable=False, name=layer._r_r_bias.name
            )

    # set final head to be trainable
    model_adapter.layers[-2].trainable = True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_adapter.layers:
        if re.match("multihead_attention", l.name):  # kv layers
            params_added += param_count(l._k_layer._ia3_layer)
            params_added += param_count(l._v_layer._ia3_layer)
        elif re.match("dense", l.name) and l.input_shape[-1] == 1536:  # ff layers
            params_added += param_count(l._ia3_layer)

    print("params added/unfrozen by ia3: %d" % params_added)

    return model_adapter


def merge_ia3(original_model, ia3_model):
    # original model contains pre-trained weights
    # ia3 model is the fine-tuned ia3 model
    for i, layer in enumerate(original_model.layers):
        # attention layers
        if re.match("multihead_attention", layer.name):
            # scale k
            k_scaler = ia3_model.layers[i]._k_layer._ia3_layer.kernel[0]
            layer._k_layer.kernel.assign(layer._k_layer.kernel * k_scaler)
            # scale v
            v_scaler = ia3_model.layers[i]._v_layer._ia3_layer.kernel[0]
            layer._v_layer.kernel.assign(layer._v_layer.kernel * v_scaler)
        # ff layers
        elif re.match("dense", layer.name) and layer.input_shape[-1] == 1536:
            ff_scaler = tf.expand_dims(ia3_model.layers[i]._ia3_layer.kernel[0], 1)
            layer.kernel.assign(layer.kernel * ff_scaler)
        # other layers
        else:
            layer.set_weights(ia3_model.layers[i].get_weights())


#############
# add locon #
#############
def add_locon(input_model, strand_pair, conv_select=None, rank=4, alpha=1):
    # first add lora to attention
    add_lora(input_model, report_param=False)

    # decide:
    # 1. whether conv1 is trainable
    # 2. which conv layers to add loRA

    # all conv layers
    conv_layers = []
    for layer in input_model.layers:
        if re.match("conv1d", layer.name):
            conv_layers += [layer.name]

    if conv_select is None:
        conv_select = len(conv_layers)

    if conv_select > len(conv_layers):
        raise ValueError(
            "conv_select must be less than number of conv layers %d." % len(conv_layers)
        )

    locon_layers = []
    conv1_tune = False
    if conv_select == len(conv_layers):
        locon_layers = conv_layers[1:]
        conv1_tune = True
    else:
        locon_layers = conv_layers[-conv_select:]

    layer_parent_dict_old = keras2dict(input_model)

    # remove switch_reverse_layer
    to_fix = [i for i in layer_parent_dict_old if re.match("switch_reverse", i)]
    for i in to_fix:
        del layer_parent_dict_old[i]

    # create new graph
    layer_output_dict_new = {}  # the output tensor of each layer in the new graph
    layer_output_dict_new.update({input_model.layers[0].name: input_model.input})

    # Iterate over all layers after the input
    reverse_bool = None
    for layer in input_model.layers[1:-1]:
        # get layer inputs
        parent_layers = layer_parent_dict_old[layer.name]
        layer_input = [layer_output_dict_new[parent] for parent in parent_layers]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # construct
        if re.match("stochastic_reverse_complement", layer.name):
            x, reverse_bool = layer(layer_input)
        elif layer.name in locon_layers:
            x = adapters.Locon(layer, trainable=True, rank=rank, alpha=alpha)(
                layer_input
            )
        else:
            x = layer(layer_input)

        # save layers to dictionary
        layer_output_dict_new.update({layer.name: x})

    final = layers.SwitchReverse(strand_pair)(
        [layer_output_dict_new[input_model.layers[-2].name], reverse_bool]
    )
    model_adapter = tf.keras.Model(inputs=input_model.inputs, outputs=final)

    if conv1_tune:
        model_adapter.get_layer(name=conv_layers[0]).trainable = True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    if conv1_tune:
        params_added += param_count(model_adapter.get_layer(name=conv_layers[0]))
    for l in model_adapter.layers:
        if re.match("multihead_attention", l.name):
            params_added += param_count(l._q_layer.down_layer)
            params_added += param_count(l._q_layer.up_layer)
            params_added += param_count(l._v_layer.down_layer)
            params_added += param_count(l._v_layer.up_layer)
        if l.name in locon_layers:
            params_added += param_count(l.down_layer)
            params_added += param_count(l.up_layer)

    print("params added/unfrozen by lora: %d" % params_added)

    return model_adapter


#### functions to merge locon
def lora_increment(layer):
    down_weights = layer.down_layer.kernel
    up_weights = layer.up_layer.kernel
    increment_weights = tf.einsum("ab,bc->ac", down_weights, up_weights) * layer.scale
    return increment_weights


def locon_increment(layer):
    down_weights = layer.down_layer.kernel
    up_weights = layer.up_layer.kernel[0]
    increment_weights = tf.einsum("abc,cd->abd", down_weights, up_weights) * layer.scale
    return increment_weights


def merge_locon(original_model, locon_model):
    # original model contains pre-trained weights
    for i, layer in enumerate(original_model.layers):
        # lora layers
        if re.match("multihead_attention", layer.name):
            q = locon_model.layers[i]._q_layer
            k = locon_model.layers[i]._k_layer
            v = locon_model.layers[i]._v_layer
            e = locon_model.layers[i]._embedding_layer
            if isinstance(q, adapters.Lora):
                increment_weights = lora_increment(q)
                layer._q_layer.kernel.assign_add(increment_weights)
            if isinstance(v, adapters.Lora):
                increment_weights = lora_increment(v)
                layer._v_layer.kernel.assign_add(increment_weights)
            if isinstance(k, adapters.Lora):
                increment_weights = lora_increment(k)
                layer._k_layer.kernel.assign_add(increment_weights)
            if isinstance(e, adapters.Lora):
                increment_weights = lora_increment(e)
                layer._embedding_layer.kernel.assign_add(increment_weights)

        # locon layers
        elif isinstance(locon_model.layers[i], adapters.Locon):
            increment_weights = locon_increment(locon_model.layers[i])
            layer.kernel.assign_add(increment_weights)

        else:
            layer.set_weights(locon_model.layers[i].get_weights())


##############
# houlsby_se #
##############
def add_houlsby_se(
    input_model, strand_pair, houlsby_latent=8, conv_select=None, se_rank=16
):
    # add squeeze-excitation blocks after conv
    # input_model should be properly frozen
    # pre_att: add se_block to pre-attention conv1d
    # all: add se_block to pre-attention conv1d and post-attention separable_conv1d

    ##################
    # houlsby layers #
    ##################
    houlsby_layers = []
    for i in range(len(input_model.layers) - 1):
        layer = input_model.layers[i]
        next_layer = input_model.layers[i + 1]
        if re.match("dropout", layer.name) and re.match("add", next_layer.name):
            houlsby_layers += [next_layer.name]

    #############
    # SE layers #
    #############
    conv_layers = []
    for layer in input_model.layers:
        if re.match("conv1d", layer.name):
            conv_layers += [layer.name]
    if conv_select is None:
        se_layers = conv_layers[1:]
    if conv_select >= len(conv_layers):
        raise ValueError(
            "conv_select must be less than number of conv layers %d." % len(conv_layers)
        )
    se_layers = conv_layers[-conv_select:]

    ###################
    # construct model #
    ###################
    layer_parent_dict_old = keras2dict(input_model)
    # remove switch_reverse_layer
    to_fix = [i for i in layer_parent_dict_old if re.match("switch_reverse", i)]
    for i in to_fix:
        del layer_parent_dict_old[i]
    # create new graph
    layer_output_dict_new = {}  # the output tensor of each layer in the new graph
    layer_output_dict_new.update({input_model.layers[0].name: input_model.input})
    # Iterate over all layers after the input
    reverse_bool = None

    for layer in input_model.layers[1:-1]:
        # parent layers
        parent_layers = layer_parent_dict_old[layer.name]

        # layer inputs
        layer_input = [layer_output_dict_new[parent] for parent in parent_layers]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if layer.name.startswith("stochastic_reverse_complement"):
            x, reverse_bool = layer(layer_input)

        # insert houlsby layer:
        elif layer.name in houlsby_layers:
            print("adapter added before:%s" % layer.name)
            x = adapters.AdapterHoulsby(latent_size=houlsby_latent)(layer_input[1])
            x = layer([layer_input[0], x])

        # insert squeeze-excite layer:
        elif layer.name in se_layers:
            se_layer = layers.SqueezeExcite(
                activation=None,  # no activation before squeezing
                additive=False,  # use sigmoid multiplicative scaling
                rank=se_rank,  # bottleneck ratio
                use_bias=False,  # ignore bias
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=1e-3
                ),  # near-zero weight initialization
                scale_fun="tanh",
            )
            x = layer(layer_input)
            x = x + se_layer(x)

        else:
            x = layer(layer_input)

        # save the output tensor of every layer
        layer_output_dict_new.update({layer.name: x})

    final = layers.SwitchReverse(strand_pair)(
        [layer_output_dict_new[input_model.layers[-2].name], reverse_bool]
    )
    model_final = tf.keras.Model(inputs=input_model.inputs, outputs=final)

    # set trainable
    for l in model_final.layers[:-2]:  # trunk
        if re.match("layer_normalization|adapter_houlsby", l.name):
            l.trainable = True
        else:
            l.trainable = False

    for l in model_final.layers:  # set trunk
        if l.name.startswith("squeeze_excite"):
            l.trainable = True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_final.layers:
        if re.match("squeeze_excite|adapter_houlsby", l.name):
            params_added += param_count(l)
        elif l.name.startswith("layer_normalization"):
            params_added += param_count(l, type="trainable")
    print("params added/unfrozen by houlsby_se: %d" % params_added)

    return model_final
