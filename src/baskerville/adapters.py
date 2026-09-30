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

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#####################
# transfer learning #
#####################
class IA3(tf.keras.layers.Layer):
    # https://arxiv.org/pdf/2205.05638.pdf
    # ia3 module for attention layer, scale output.

    def __init__(self, original_layer, trainable=False, **kwargs):
        # keep the name of this layer the same as the original dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.output_dim = original_layer_config["units"]

        self.original_layer = original_layer
        self.original_layer.trainable = False

        # IA3 weights. Make it a dense layer to control trainable
        self._ia3_layer = tf.keras.layers.Dense(
            units=self.output_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
            trainable=True,
            name="ia3",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        scaler = self._ia3_layer(tf.constant([[1]], dtype="float64"))[0]
        return original_output * scaler

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "size": self.output_dim,
            }
        )
        return config


class IA3_ff(tf.keras.layers.Layer):
    # https://arxiv.org/pdf/2205.05638.pdf
    # ia3 module for down-projection ff layer, scale input.

    def __init__(self, original_layer, trainable=False, **kwargs):
        # keep the name of this layer the same as the original dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.input_dim = original_layer.input_shape[-1]

        self.original_layer = original_layer
        self.original_layer.trainable = False

        # IA3 weights. Make it a dense layer to control trainable
        self._ia3_layer = tf.keras.layers.Dense(
            units=self.input_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
            trainable=True,
            name="ia3_ff",
        )

    def call(self, inputs):
        scaler = self._ia3_layer(tf.constant([[1]], dtype="float64"))[0]
        return self.original_layer(inputs * scaler)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"size": self.input_dim})
        return config


class Lora(tf.keras.layers.Layer):
    # adapted from:
    # https://arxiv.org/abs/2106.09685
    # https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/
    # https://github.com/Elvenson/stable-diffusion-keras-ft/blob/main/layers.py

    def __init__(self, original_layer, rank=8, alpha=16, trainable=False, **kwargs):
        # keep the name of this layer the same as the original dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.output_dim = original_layer_config["units"]

        if rank > self.output_dim:
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {self.output_dim}"
            )

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.original_layer = original_layer
        self.original_layer.trainable = False

        # Note: the original paper mentions that normal distribution was
        # used for initialization. However, the official LoRA implementation
        # uses "Kaiming/He Initialization".
        self.down_layer = tf.keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            # kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1 / self.rank),
            trainable=True,
            name="lora_a",
        )

        self.up_layer = tf.keras.layers.Dense(
            units=self.output_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="lora_b",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        lora_output = self.up_layer(self.down_layer(inputs)) * self.scale
        return original_output + lora_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"rank": self.rank, "alpha": self.alpha})
        return config


class Locon(tf.keras.layers.Layer):
    # LoRA for conv-layer, adapted from:
    # https://arxiv.org/pdf/2309.14859#page=23.84
    # https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/locon.py
    # use default alpha and rank for locon

    def __init__(self, original_layer, rank=4, alpha=1, trainable=False, **kwargs):
        # keep the name of this layer the same as the original conv layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.input_dim = original_layer.input_shape[-1]
        self.output_dim = original_layer_config["filters"]

        if rank > self.output_dim:
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {self.output_dim}"
            )

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.original_layer = original_layer
        self.original_layer.trainable = False

        output_dim = original_layer_config["filters"]
        kernel_size = original_layer_config["kernel_size"][0]
        stride = original_layer_config["strides"][0]
        dilation_rate = original_layer_config["dilation_rate"][0]

        # Note: the original paper mentions that normal distribution was
        # used for initialization. However, the official LoRA implementation
        # uses "Kaiming/He Initialization".

        self.down_layer = tf.keras.layers.Conv1D(
            filters=rank,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
            dilation_rate=dilation_rate,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            name="locon_down",
        )

        self.up_layer = tf.keras.layers.Conv1D(
            filters=output_dim,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="locon_up",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        lora_output = self.up_layer(self.down_layer(inputs)) * self.scale
        return original_output + lora_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"rank": self.rank, "alpha": self.alpha})
        return config


class AdapterHoulsby(tf.keras.layers.Layer):
    # https://arxiv.org/abs/1902.00751
    # adapted from: https://github.com/jain-harshil/Adapter-BERT

    def __init__(self, latent_size, activation=tf.keras.layers.ReLU(), **kwargs):
        super(AdapterHoulsby, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.activation = activation

    def build(self, input_shape):
        self.down_project = tf.keras.layers.Dense(
            units=self.latent_size,
            activation="linear",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
            bias_initializer="zeros",
            name="adapter_down",
        )

        self.up_project = tf.keras.layers.Dense(
            units=input_shape[-1],
            activation="linear",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
            bias_initializer="zeros",
            name="adapter_up",
        )

    def call(self, inputs):
        projected_down = self.down_project(inputs)
        activated = self.activation(projected_down)
        projected_up = self.up_project(activated)
        output = projected_up + inputs
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"latent_size": self.latent_size, "activation": self.activation})
        return config
