# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import sys

import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

############################################################
# Basic
############################################################


class Scale(tf.keras.layers.Layer):
    """Scale the input by a learned value.

    Args:
        axis (int or [int]): Axis/axes along which to scale.
        initializer: Initializer for the scale weight.
    """

    def __init__(self, axis=-1, initializer="zeros"):
        super(Scale, self).__init__()
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError("Input has undefined rank.")
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        elif isinstance(self.axis, tuple):
            self.axis = list(self.axis)
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError("Invalid axis: %d" % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError("Duplicate axis: {}".format(tuple(self.axis)))

        param_shape = [input_shape[dim] for dim in self.axis]

        self.scale = self.add_weight(
            name="scale",
            shape=param_shape,
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, x):
        return x * self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "axis": self.axis,
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config


class Softplus(tf.keras.layers.Layer):
    """Safe softplus, clipping large values."""

    def __init__(self, exp_max=10000):
        super(Softplus, self).__init__()
        self.exp_max = exp_max

    def call(self, x):
        x = tf.clip_by_value(x, -self.exp_max, self.exp_max)
        return tf.keras.activations.softplus(x)

    def get_config(self):
        config = super().get_config().copy()
        config["exp_max"] = self.exp_max
        return config


############################################################
# Center ops
############################################################


class CenterSlice(tf.keras.layers.Layer):
    """Scale the input by a learned value.

    Args:
        axis (int or [int]): Axis/axes along which to scale.
        initializer: Initializer for the scale weight.
    """

    def __init__(self, center):
        super(CenterSlice, self).__init__()
        self.center = center

    def call(self, x):
        seq_len = x.shape[1]
        center_start = (seq_len - self.center) // 2
        center_end = center_start + self.center
        return x[:, center_start:center_end, :]

    def get_config(self):
        config = super().get_config().copy()
        config.update({"center": self.center})
        return config


class CenterAverage(tf.keras.layers.Layer):
    """Average the center of the input.

    Args:
        center (int): Length of the center slice.
    """

    def __init__(self, center):
        super(CenterAverage, self).__init__()
        self.center = center
        self.slice = CenterSlice(self.center)

    def call(self, x):
        return tf.keras.backend.mean(self.slice(x), axis=1, keepdims=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"center": self.center})
        return config


class LengthAverage(tf.keras.layers.Layer):
    """Average across a variable length sequence."""

    def __init__(self):
        super(LengthAverage, self).__init__()

    def call(self, x, seq):
        # focus on nt's
        seq = seq[..., :4]

        # collapse nt axis
        seq_pos = tf.math.reduce_sum(seq, axis=-1)

        # collapse length axis
        seq_len = tf.math.reduce_sum(seq_pos, axis=-1)
        seq_len = tf.expand_dims(seq_len, axis=-1)

        # sum across length
        x_sum = tf.math.reduce_mean(x, axis=-2)

        # divide by true sequence length
        x_avg = x_sum / seq_len

        return x_avg


############################################################
# Attention
############################################################


def _prepend_dims(x, num_dims):
    return tf.reshape(x, shape=[1] * num_dims + x.shape)


def positional_features_central_mask(
    positions: tf.Tensor, feature_size: int, seq_length: int
):
    """Positional features using a central mask (allow only central features)."""
    pow_rate = np.exp(np.log(seq_length + 1) / feature_size).astype("float32")
    center_widths = tf.pow(pow_rate, tf.range(1, feature_size + 1, dtype=tf.float32))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def positional_features(
    positions: tf.Tensor, feature_size: int, seq_length: int, symmetric=False
):
    """Compute relative positional encodings/features.

    Each positional feature function will compute/provide the same fraction of
    features, making up the total of feature_size.

    Args:
      positions: Tensor of relative positions of arbitrary shape.
      feature_size: Total number of basis functions.
      seq_length: Sequence length denoting the characteristic length that
        the individual positional features can use. This is required since the
        parametrization of the input features should be independent of `positions`
        while it could still require to use the total number of features.
      symmetric: If True, the resulting features will be symmetric across the
        relative position of 0 (i.e. only absolute value of positions will
        matter). If false, then both the symmetric and asymmetric version
        (symmetric multiplied by sign(positions)) of the features will be used.

    Returns:
      Tensor of shape: `positions.shape + (feature_size,)`.
    """
    if symmetric:
        num_components = 1
    else:
        num_components = 2
    num_basis_per_class = feature_size // num_components

    embeddings = positional_features_central_mask(
        positions, num_basis_per_class, seq_length
    )

    if not symmetric:
        embeddings = tf.concat(
            [embeddings, tf.sign(positions)[..., tf.newaxis] * embeddings], axis=-1
        )

    tf.TensorShape(embeddings.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )

    return embeddings


def relative_shift(x):
    """Shift the relative logits like in TransformerXL."""
    # We prepend zeros on the final timescale dimension.
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


class MultiheadAttention(tf.keras.layers.Layer):
    """Multi-head attention."""

    def __init__(
        self,
        value_size,
        key_size,
        heads,
        scaling=True,
        attention_dropout_rate=0,
        relative_position_symmetric=False,
        relative_position_functions=["positional_features_central_mask"],
        num_position_features=None,
        positional_dropout_rate=0,
        content_position_bias=True,
        zero_initialize=True,
        transpose_stride=0,
        gated=False,
        initializer="he_normal",
        l2_scale=0,
        qkv_width=1,
        seqlen_train=None,
    ):
        """Creates a MultiheadAttention module.
           Original version written by Ziga Avsec.

        Args:
          value_size: The size of each value embedding per head.
          key_size: The size of each key and query embedding per head.
          heads: The number of independent queries per timestep.
          scaling: Whether to scale the attention logits.
          attention_dropout_rate: Dropout rate for attention logits.
          relative_position_symmetric: If True, the symmetric version of basis
            functions will be used. If False, a symmetric and asymmetric versions
            will be use.
          relative_position_functions: List of function names used for relative
            positional biases.
          num_position_features: Number of relative positional features
            to compute. If None, `value_size * num_heads` is used.
          positional_dropout_rate: Dropout rate for the positional encodings if
            relative positions are used.
          zero_initialize: if True, the final linear layer will be 0 initialized.
          initializer: Initializer for the projection layers. If unspecified,
            VarianceScaling is used with scale = 2.0.
        """
        super().__init__()
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._gated = gated
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        self.seqlen_train = seqlen_train
        if num_position_features is None:
            # num_position_features needs to be divisible by the number of
            # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_position_features = (
                self._value_size // divisible_by
            ) * divisible_by
        else:
            self._num_position_features = num_position_features
        self._positional_dropout_rate = positional_dropout_rate
        self._content_position_bias = content_position_bias
        self._l2_scale = l2_scale
        self._initializer = initializer

        key_proj_size = self._key_size * self._num_heads
        embedding_size = self._value_size * self._num_heads

        if qkv_width == 1:
            # standard dense layers
            self._q_layer = tf.keras.layers.Dense(
                key_proj_size,
                name="q_layer",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._k_layer = tf.keras.layers.Dense(
                key_proj_size,
                name="k_layer",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._v_layer = tf.keras.layers.Dense(
                embedding_size,
                name="v_layer",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
        else:
            # CvT separable convolutions
            self._q_layer = tf.keras.layers.SeparableConv1D(
                key_proj_size,
                kernel_size=qkv_width,
                padding="same",
                name="q_layer",
                use_bias=False,
                depthwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._k_layer = tf.keras.layers.SeparableConv1D(
                key_proj_size,
                kernel_size=qkv_width,
                padding="same",
                name="k_layer",
                use_bias=False,
                depthwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._v_layer = tf.keras.layers.SeparableConv1D(
                embedding_size,
                kernel_size=qkv_width,
                padding="same",
                name="v_layer",
                use_bias=False,
                depthwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )

        if self._gated:
            self._gate_layer = tf.keras.layers.Dense(
                embedding_size,
                activation="activation",
                name="gate",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
        w_init = tf.keras.initializers.Zeros() if zero_initialize else self._initializer
        if transpose_stride > 0:
            self._embedding_layer = tf.keras.layers.Conv1DTranspose(
                filters=embedding_size,
                kernel_size=3,
                strides=transpose_stride,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )
        else:
            self._embedding_layer = tf.keras.layers.Dense(
                embedding_size,
                name="embedding_layer",
                kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )

        # Create relative position layers
        self._r_k_layer = tf.keras.layers.Dense(
            key_proj_size,
            name="r_k_layer",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
            kernel_initializer=self._initializer,
        )
        self._r_w_bias = self.add_weight(
            "%s/r_w_bias" % self.name,
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype=tf.float32,
            trainable=True,
        )
        self._r_r_bias = self.add_weight(
            "%s/r_r_bias" % self.name,
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype=tf.float32,
            trainable=True,
        )

    def _multihead_output(self, linear_layer, inputs):
        """Applies a standard linear to inputs and returns multihead output."""
        output = linear_layer(inputs)  # [B, T, H * KV]
        _, seq_len, num_channels = output.shape

        # Split H * Channels into separate axes.
        num_kv_channels = num_channels // self._num_heads
        output = tf.reshape(
            output, shape=[-1, seq_len, self._num_heads, num_kv_channels]
        )
        # [B, T, H, KV] -> [B, H, T, KV]
        return tf.transpose(output, [0, 2, 1, 3])

    def call(self, inputs, training=False):
        # Initialise the projection layers.
        embedding_size = self._value_size * self._num_heads
        seq_len = inputs.shape[1]

        # Compute q, k and v as multi-headed projections of the inputs.
        q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
        k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
        v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size**-0.5

        # [B, H, T', T]
        # content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)
        content_logits = tf.matmul(
            q + tf.cast(self._r_w_bias, dtype=inputs.dtype), k, transpose_b=True
        )

        if self._num_position_features == 0:
            logits = content_logits
        else:
            # Project positions to form relative keys.
            distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]

            if self.seqlen_train is None:
                positional_encodings = positional_features(
                    positions=distances,
                    feature_size=self._num_position_features,
                    seq_length=seq_len,
                    symmetric=self._relative_position_symmetric,
                )
                # [1, 2T-1, Cr]
            else:
                positional_encodings = positional_features(
                    positions=distances,
                    feature_size=self._num_position_features,
                    seq_length=self.seqlen_train,
                    symmetric=self._relative_position_symmetric,
                )
                # [1, 2T-1, Cr]

            if training:
                positional_encodings = tf.nn.dropout(
                    positional_encodings, rate=self._positional_dropout_rate
                )

            # [1, H, 2T-1, K]
            r_k = self._multihead_output(self._r_k_layer, positional_encodings)

            # Add shifted relative logits to content logits.
            if self._content_position_bias:
                # [B, H, T', 2T-1]
                # relative_logits = tf.matmul(q + self._r_r_bias, r_k, transpose_b=True)
                relative_logits = tf.matmul(
                    q + tf.cast(self._r_r_bias, dtype=inputs.dtype),
                    r_k,
                    transpose_b=True,
                )
            else:
                # [1, H, 1, 2T-1]
                # relative_logits = tf.matmul(self._r_r_bias, r_k, transpose_b=True)
                relative_logits = tf.matmul(
                    tf.cast(self._r_r_bias, dtype=inputs.dtype), r_k, transpose_b=True
                )
                # [1, H, T', 2T-1]
                relative_logits = tf.broadcast_to(
                    relative_logits,
                    shape=(1, self._num_heads, seq_len, 2 * seq_len - 1),
                )

            #  [B, H, T', T]
            relative_logits = relative_shift(relative_logits)
            logits = content_logits + relative_logits

        # softmax across length
        weights = tf.nn.softmax(logits)

        # Dropout on the attention weights.
        if training:
            weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

        # Transpose and reshape the output.
        output = tf.matmul(weights, v)  # [B, H, T', V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]
        attended_inputs = tf.reshape(
            output_transpose, shape=[-1, seq_len, embedding_size]
        )

        # Gate
        if self._gated:
            attended_inputs = self._gate_layer(attended_inputs)

        # Final linear layer
        output = self._embedding_layer(attended_inputs)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"value_size": self._value_size, "key_size": self._key_size})
        return config


class WheezeExcite(tf.keras.layers.Layer):
    def __init__(self, pool_size):
        super(WheezeExcite, self).__init__()
        self.pool_size = pool_size
        assert self.pool_size % 2 == 1
        self.paddings = [[0, 0], [self.pool_size // 2, self.pool_size // 2], [0, 0]]

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

        self.wheeze = tf.keras.layers.AveragePooling1D(
            self.pool_size, strides=1, padding="valid"
        )

        self.excite1 = tf.keras.layers.Dense(
            units=self.num_channels // 4, activation="relu"
        )
        self.excite2 = tf.keras.layers.Dense(units=self.num_channels, activation="relu")

    def call(self, x):
        # pad
        x_pad = tf.pad(x, self.paddings, "SYMMETRIC")

        # squeeze
        x_squeeze = self.wheeze(x_pad)

        # excite
        x_excite = self.excite1(x_squeeze)
        x_excite = self.excite2(x_excite)
        x_excite = tf.keras.activations.sigmoid(x_excite)

        # scale
        xs = x * x_excite

        return xs

    def get_config(self):
        config = super().get_config().copy()
        config.update({"pool_size": self.pool_size})
        return config


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(
        self,
        activation="relu",
        additive=False,
        rank=8,
        norm_type=None,
        bn_momentum=0.9,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        scale_fun="sigmoid",
    ):
        super(SqueezeExcite, self).__init__()
        self.activation = activation
        self.additive = additive
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum
        self.rank = rank
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.scale_fun = scale_fun

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

        if len(input_shape) == 3:
            self.one_or_two = "one"
            self.gap = tf.keras.layers.GlobalAveragePooling1D()
        elif len(input_shape) == 4:
            self.one_or_two = "two"
            self.gap = tf.keras.layers.GlobalAveragePooling2D()
        else:
            print(
                "SqueezeExcite: input dim %d unexpected" % len(input_shape),
                file=sys.stderr,
            )
            exit(1)

        if self.scale_fun == "sigmoid":
            self.scale_f = tf.keras.activations.sigmoid
        elif self.scale_fun == "tanh":  # set to tanh for transfer
            self.scale_f = tf.keras.activations.tanh
        else:
            print(
                "scale function must be sigmoid or tanh",
                file=sys.stderr,
            )
            exit(1)

        self.dense1 = tf.keras.layers.Dense(
            units=self.rank,
            activation="relu",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self.dense2 = tf.keras.layers.Dense(
            units=self.num_channels,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            activation=None,
        )

    def call(self, x):
        # activate
        if self.activation is not None:
            x = activate(x, self.activation)

        # squeeze
        squeeze = self.gap(x)

        # excite
        excite = self.dense1(squeeze)
        excite = self.dense2(excite)

        # scale
        if self.one_or_two == "one":
            excite = tf.reshape(excite, [-1, 1, self.num_channels])
        else:
            excite = tf.reshape(excite, [-1, 1, 1, self.num_channels])

        if self.additive:
            xs = x + excite
        else:
            excite = self.scale_f(excite)
            xs = x * excite

        return xs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "activation": self.activation,
                "additive": self.additive,
                "use_bias": self.use_bias,
                "norm_type": self.norm_type,
                "bn_momentum": self.bn_momentum,
                "rank": self.rank,
            }
        )
        return config


class GlobalContext(tf.keras.layers.Layer):
    def __init__(self):
        super(GlobalContext, self).__init__()

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

        self.context_key = tf.keras.layers.Dense(units=1, activation=None)

        self.dense1 = tf.keras.layers.Dense(units=self.num_channels // 4)
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(units=self.num_channels)

    def call(self, x):
        # context attention
        keys = self.context_key(x)  # [batch x length x 1]
        attention = tf.keras.activations.softmax(keys, axis=-2)  # [batch x length x 1]

        # context summary
        context = x * attention  # [batch x length x channels]
        context = tf.keras.backend.sum(
            context, axis=-2, keepdims=True
        )  # [batch x 1 x channels]

        # transform
        transform = self.dense1(context)  # [batch x 1 x channels/4]
        transform = tf.keras.activations.relu(
            self.ln(transform)
        )  # [batch x 1 x channels/4]
        transform = self.dense2(transform)  # [batch x 1 x channels]
        # transform = tf.reshape(transform, [-1,1,self.num_channels])

        # fusion
        xs = x + transform  # [batch x length x channels]

        return xs


############################################################
# Pooling
############################################################
class SoftmaxPool1D(tf.keras.layers.Layer):
    """Pooling operation with optional weights."""

    def __init__(
        self, pool_size: int = 2, per_channel: bool = False, init_gain: float = 2.0
    ):
        """Softmax pooling.

        Args:
          pool_size: Pooling size, same as in Max/AvgPooling.
          per_channel: If True, the logits/softmax weights will be computed for
            each channel separately. If False, same weights will be used across all
            channels.
          init_gain: When 0.0 is equivalent to avg pooling, and when
            ~2.0 and it's equivalent to max pooling.
        """
        super(SoftmaxPool1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.init_gain = init_gain
        self.logit_linear = None

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.logit_linear = tf.keras.layers.Dense(
            units=self.num_channels if self.per_channel else 1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Identity(self.init_gain),
        )

    def call(self, inputs):
        _, seq_length, num_channels = inputs.shape
        inputs = tf.reshape(
            inputs, (-1, seq_length // self.pool_size, self.pool_size, num_channels)
        )
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self.logit_linear(inputs), axis=-2), axis=-2
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({"pool_size": self.pool_size, "init_gain": self.init_gain})
        return config


############################################################
# Position
############################################################
class ConcatPosition(tf.keras.layers.Layer):
    """Concatenate position to 1d feature vectors."""

    def __init__(self, transform=None, power=1):
        super(ConcatPosition, self).__init__()
        self.transform = transform
        self.power = power

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        pos_range = tf.range(-seq_len // 2, seq_len // 2)
        if self.transform is None:
            pos_feature = pos_range
        elif self.transform == "abs":
            pos_feature = tf.math.abs(pos_range)
        elif self.transform == "reversed":
            pos_feature = pos_range[::-1]
        else:
            raise ValueError("Unknown ConcatPosition transform.")

        if self.power != 1:
            pos_feature = tf.pow(pos_feature, self.power)
        pos_feature = tf.expand_dims(pos_feature, axis=0)
        pos_feature = tf.expand_dims(pos_feature, axis=-1)
        pos_feature = tf.tile(pos_feature, [batch_size, 1, 1])
        pos_feature = tf.dtypes.cast(pos_feature, dtype=tf.float32)

        return tf.concat([pos_feature, inputs], axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"transform": self.transform, "power": self.power})
        return config


############################################################
# 2D
############################################################
class OneToTwo(tf.keras.layers.Layer):
    """Transform 1d to 2d with i,j vectors operated on."""

    def __init__(self, operation="mean"):
        super(OneToTwo, self).__init__()
        self.operation = operation.lower()
        valid_operations = ["concat", "mean", "max", "multipy", "multiply1"]
        assert self.operation in valid_operations

    def call(self, oned):
        _, seq_len, features = oned.shape

        twod1 = tf.tile(oned, [1, seq_len, 1])
        twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
        twod2 = tf.transpose(twod1, [0, 2, 1, 3])

        if self.operation == "concat":
            twod = tf.concat([twod1, twod2], axis=-1)

        elif self.operation == "multiply":
            twod = tf.multiply(twod1, twod2)

        elif self.operation == "multiply1":
            twod = tf.multiply(twod1 + 1, twod2 + 1) - 1

        else:
            twod1 = tf.expand_dims(twod1, axis=-1)
            twod2 = tf.expand_dims(twod2, axis=-1)
            twod = tf.concat([twod1, twod2], axis=-1)

            if self.operation == "mean":
                twod = tf.reduce_mean(twod, axis=-1)

            elif self.operation == "max":
                twod = tf.reduce_max(twod, axis=-1)

        return twod

    def get_config(self):
        config = super().get_config().copy()
        config["operation"] = self.operation
        return config


class ConcatDist2D(tf.keras.layers.Layer):
    """Concatenate the pairwise distance to 2d feature matrix."""

    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        ## concat 2D distance ##
        pos = tf.expand_dims(tf.range(0, seq_len), axis=-1)
        matrix_repr1 = tf.tile(pos, [1, seq_len])
        matrix_repr2 = tf.transpose(matrix_repr1, [1, 0])
        dist = tf.math.abs(tf.math.subtract(matrix_repr1, matrix_repr2))
        dist = tf.dtypes.cast(dist, tf.float32)
        dist = tf.expand_dims(dist, axis=-1)
        dist = tf.expand_dims(dist, axis=0)
        dist = tf.tile(dist, [batch_size, 1, 1, 1])
        return tf.concat([inputs, dist], axis=-1)


class UpperTri(tf.keras.layers.Layer):
    """Unroll matrix to its upper triangular portion."""

    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, inputs):
        seq_len = inputs.shape[1]
        output_dim = inputs.shape[-1]

        if isinstance(seq_len, tf.compat.v1.Dimension):
            seq_len = seq_len.value
            output_dim = output_dim.value

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0] + seq_len * triu_tup[1])
        unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
        return tf.gather(unroll_repr, triu_index, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config["diagonal_offset"] = self.diagonal_offset
        return config


class Symmetrize2D(tf.keras.layers.Layer):
    """Take the average of a matrix and its transpose to enforce symmetry."""

    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def call(self, x):
        x_t = tf.transpose(x, [0, 2, 1, 3])
        x_sym = (x + x_t) / 2
        return x_sym


############################################################
# Augmentation
############################################################


class EnsembleReverseComplement(tf.keras.layers.Layer):
    """Expand tensor to include reverse complement of one hot encoded DNA sequence."""

    def __init__(self):
        super(EnsembleReverseComplement, self).__init__()

    def call(self, seqs_1hot):
        if not isinstance(seqs_1hot, list):
            seqs_1hot = [seqs_1hot]

        ens_seqs_1hot = []
        for seq_1hot in seqs_1hot:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            ens_seqs_1hot += [
                (seq_1hot, tf.constant(False)),
                (rc_seq_1hot, tf.constant(True)),
            ]

        return ens_seqs_1hot


class StochasticReverseComplement(tf.keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


class SwitchReverse(tf.keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self, strand_pair=None):
        super(SwitchReverse, self).__init__()
        self.strand_pair = strand_pair

    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 2:
            # because we collapsed length already
            rev_axes = []
        elif xd == 3:
            # length axis
            rev_axes = [1]
        elif xd == 4:
            # 2d spatial axes
            rev_axes = [1, 2]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)

        if len(rev_axes) > 0:
            xr = tf.keras.backend.switch(reverse, tf.reverse(x, axis=rev_axes), x)
        else:
            xr = x

        if self.strand_pair is None:
            xrs = xr
        else:
            xrs = tf.keras.backend.switch(
                reverse, tf.gather(xr, self.strand_pair, axis=-1), xr
            )

        return xrs

    def get_config(self):
        config = super().get_config().copy()
        config["strand_pair"] = self.strand_pair
        return config


class SwitchReverseTriu(tf.keras.layers.Layer):
    def __init__(self, diagonal_offset):
        super(SwitchReverseTriu, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, x_reverse):
        x_ut = x_reverse[0]
        reverse = x_reverse[1]

        # infer original sequence length
        ut_len = x_ut.shape[1]
        if isinstance(ut_len, tf.compat.v1.Dimension):
            ut_len = ut_len.value
        seq_len = int(np.sqrt(2 * ut_len + 0.25) - 0.5)
        seq_len += self.diagonal_offset

        # get triu indexes
        ut_indexes = np.triu_indices(seq_len, self.diagonal_offset)
        assert len(ut_indexes[0]) == ut_len

        # construct a ut matrix of ut indexes
        mat_ut_indexes = np.zeros(shape=(seq_len, seq_len), dtype="int")
        mat_ut_indexes[ut_indexes] = np.arange(ut_len)

        # make lower diag mask
        mask_ut = np.zeros(shape=(seq_len, seq_len), dtype="bool")
        mask_ut[ut_indexes] = True
        mask_ld = ~mask_ut

        # construct a matrix of symmetric ut indexes
        mat_indexes = mat_ut_indexes + np.multiply(mask_ld, mat_ut_indexes.T)

        # reverse complement
        mat_rc_indexes = mat_indexes[::-1, ::-1]

        # extract ut order
        rc_ut_order = mat_rc_indexes[ut_indexes]

        return tf.keras.backend.switch(
            reverse, tf.gather(x_ut, rc_ut_order, axis=1), x_ut
        )

    def get_config(self):
        config = super().get_config().copy()
        config["diagonal_offset"] = self.diagonal_offset
        return config


class EnsembleShift(tf.keras.layers.Layer):
    """Expand tensor to include shifts of one hot encoded DNA sequence."""

    def __init__(self, shifts=[0], pad="uniform"):
        super(EnsembleShift, self).__init__()
        self.shifts = shifts
        self.pad = pad

    def call(self, seqs_1hot):
        if not isinstance(seqs_1hot, list):
            seqs_1hot = [seqs_1hot]

        ens_seqs_1hot = []
        for seq_1hot in seqs_1hot:
            for shift in self.shifts:
                ens_seqs_1hot.append(shift_sequence(seq_1hot, shift))

        return ens_seqs_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shifts": self.shifts, "pad": self.pad})
        return config


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, symmetric=True, pad="uniform"):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(
                shape=[], minval=0, dtype=tf.int64, maxval=len(self.augment_shifts)
            )
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(
                tf.not_equal(shift, 0),
                lambda: shift_sequence(seq_1hot, shift),
                lambda: seq_1hot,
            )
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"shift_max": self.shift_max, "symmetric": self.symmetric, "pad": self.pad}
        )
        return config


def shift_sequence(seq, shift, pad_value=0):
    """Shift a sequence left or right by shift_amount.

    Args:
        seq: [batch_size, seq_length, seq_depth] sequence
        shift: signed shift value (tf.int32 or int)
        pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0 : tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(
        tf.greater(shift, 0), lambda: _shift_right(seq), lambda: _shift_left(seq)
    )
    sseq.set_shape(input_shape)

    return sseq


############################################################
# Factorization
############################################################


class FactorInverse(tf.keras.layers.Layer):
    """Inverse a target matrix factorization."""

    def __init__(self, components_npy):
        super(FactorInverse, self).__init__()
        self.components_npy = components_npy
        self.components = tf.constant(np.load(components_npy), dtype=tf.float32)

    def call(self, W):
        return tf.keras.backend.dot(W, self.components)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"components_npy": self.components_npy})
        return config


############################################################
# helpers
############################################################


def activate(current, activation, verbose=False):
    if verbose:
        print("activate:", activation)
    if activation == "relu":
        current = tf.keras.layers.ReLU()(current)
    elif activation == "gelu":
        current = tf.keras.activations.gelu(current, approximate=True)
    elif activation == "sigmoid":
        current = tf.keras.activations.sigmoid(current)
    elif activation == "tanh":
        current = tf.keras.activations.tanh(current)
    elif activation == "softplus":
        current = Softplus()(current)
    elif activation == "linear" or activation is None:
        pass
    else:
        print('Unrecognized activation "%s"' % activation, file=sys.stderr)
        exit(1)

    return current
