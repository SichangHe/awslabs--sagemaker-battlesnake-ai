# Slightly modified version of https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py from the Ray RLlib project
#
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class VisionNetwork(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(VisionNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        map_height = model_config["custom_options"]["max_map_height"]
        activation = get_activation_fn(model_config.get("conv_activation"))

        filters = model_config.get("conv_filters")

        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        input_shape = [map_height, map_height, 6]
        inputs = tf.keras.layers.Input(shape=input_shape, name="observations")
        last_layer = inputs

        # Build the action layers
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv{}".format(i),
            )(last_layer)
        out_size, kernel, stride = filters[-1]
        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            last_layer = tf.keras.layers.Conv2D(
                num_outputs,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv_out",
            )(last_layer)
            conv_out = last_layer
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv{}".format(i + 1),
            )(last_layer)
            conv_out = tf.keras.layers.Conv2D(
                num_outputs, [1, 1], activation=None, padding="same", name="conv_out"
            )(last_layer)

        # Build the value layers
        if vf_share_layers:
            last_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(
                last_layer
            )
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01),
            )(last_layer)
        else:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=(stride, stride),
                    activation=activation,
                    padding="valid",
                    name="conv_value_{}".format(i),
                )(last_layer)
            out_size, kernel, stride = filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv_value_{}".format(i + 1),
            )(last_layer)
            last_layer = tf.keras.layers.Conv2D(
                1, [1, 1], activation=None, padding="same", name="conv_value_out"
            )(last_layer)
            value_out = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(
                last_layer
            )

        self.base_model = tf.keras.Model(inputs, [conv_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        model_out, self._value_out = self.base_model(
            tf.cast(input_dict["obs"]["state"], tf.float32)
        )

        model_out = tf.squeeze(model_out, axis=[1, 2])
        mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.log(mask), tf.float32.min)
        masked_logits = inf_mask + model_out
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
