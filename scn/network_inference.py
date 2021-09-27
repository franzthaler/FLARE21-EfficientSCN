import tensorflow as tf
from tensorflow.keras.layers import Conv3D, AveragePooling3D, AlphaDropout, Activation, Dropout, Lambda
from tensorflow.keras.regularizers import l2

from tensorflow_train_v2.networks.unet_base import UnetBase
from tensorflow_train_v2.layers.layers import Sequential, ConcatChannels, UpSampling3DLinear, UpSampling3DCubic
from tensorflow_train_v2.layers.initializers import he_initializer, selu_initializer
from tensorflow_train_v2.utils.data_format import get_channel_index
from tensorflow_train_v2.layers.salt_pepper_noise import salt_pepper_3D


class UnetAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(UnetAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling3D([2] * 3, data_format=self.data_format)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling3DLinear([2] * 3, data_format=self.data_format)

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level))

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='expanding' + str(current_level))

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv3D(filters=self.num_filters_base,
                      kernel_size=self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)


def activation_fn_output_kernel_initializer(activation, data_format):
    if activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
        kernel_initializer = he_initializer
    elif activation == 'softmax':
        activation_fn = lambda x: tf.nn.softmax(x, axis=1 if data_format == 'channels_first' else -1)
        kernel_initializer = he_initializer
    else:
        activation_fn = None
        kernel_initializer = None
    return activation_fn, kernel_initializer


def activation_fn_kernel_initializer_alpha_dropout(activation):
    if activation == 'relu':
        activation_fn = tf.nn.relu
        kernel_initializer = he_initializer
        alpha_dropout = False
    elif activation == 'lrelu':
        activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        kernel_initializer = he_initializer
        alpha_dropout = False
    else:
        activation_fn = None
        kernel_initializer = None
        alpha_dropout = False
    return activation_fn, kernel_initializer, alpha_dropout


class LocalAppearance(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self,
                 num_labels,
                 activation='relu',
                 data_format='channels_first',
                 padding='same',
                 output_background=False,
                 prediction_activation='none',
                 unet_parameters=None):
        """
        The spatial configuration net.
        :param input: Input tensor.
        :param num_labels: Number of outputs.
        :param is_training: True, if training network.
        :param data_format: 'channels_first' or 'channels_last'
        :param padding: Padding parameter passed to the convolution operations.
        :param spatial_downsample: Downsamping factor for the spatial configuration stage.
        :param args: Not used.
        :param kwargs: Not used.
        :return: heatmaps, local_heatmaps, spatial_heatmaps
        """
        super(LocalAppearance, self).__init__()
        self.output_background = output_background
        self.num_labels = num_labels
        self.num_output_labels = self.num_labels - int(not self.output_background)
        self.padding = padding
        self.data_format = data_format
        self.activation_fn, self.kernel_initializer, self.alpha_dropout = activation_fn_kernel_initializer_alpha_dropout(activation)
        self.output_activation_fn, self.output_layer_kernel_initializer = activation_fn_output_kernel_initializer(prediction_activation, self.data_format)

        self.unet = UnetAvgLinear3D(data_format=self.data_format, padding=self.padding, kernel_initializer=self.kernel_initializer, activation=self.activation_fn, alpha_dropout=self.alpha_dropout, name='unet', **unet_parameters)
        self.prediction = Conv3D(self.num_output_labels, [1] * 3, name='prediction', kernel_initializer=self.output_layer_kernel_initializer, activation=None, data_format=self.data_format, padding=self.padding)
        self.prediction_activation = Activation(self.output_activation_fn, dtype='float16', name='prediction_activation')

    def call(self, inputs, training, **kwargs):
        node = self.unet(inputs, training=training)
        prediction_wo_activation = node = self.prediction(node, training=training)
        prediction = self.prediction_activation(node, training=training)
        return prediction, prediction_wo_activation


class SpatialConfiguration(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self,
                 num_labels,
                 activation='relu',
                 data_format='channels_first',
                 padding='same',
                 downsampling_factor=4,
                 local_channel_dropout_ratio=0.0,
                 image_channel_dropout_ratio=0.0,
                 input_background=False,
                 output_background=False,
                 image_as_input=False,
                 prediction_activation='none',
                 salt_pepper_ratio=0.0,
                 salt_pepper_scales=[1],
                 output_background_local=False,
                 unet_parameters=None):
        """
        The spatial configuration net.
        :param input: Input tensor.
        :param num_labels: Number of outputs.
        :param is_training: True, if training network.
        :param data_format: 'channels_first' or 'channels_last'
        :param padding: Padding parameter passed to the convolution operations.
        :param spatial_downsample: Downsamping factor for the spatial configuration stage.
        :param salt_pepper_ratio: Parameter of additional salt and pepper noise.
        :param salt_pepper_scales: Parameter of additional salt and pepper noise.
        :param args: Not used.
        :param kwargs: Not used.
        :return: heatmaps, local_heatmaps, spatial_heatmaps
        """
        super(SpatialConfiguration, self).__init__()
        self.input_background = input_background
        self.output_background = output_background
        self.image_as_input = image_as_input
        self.num_labels = num_labels
        self.padding = padding
        self.downsampling_factor = downsampling_factor
        self.local_channel_dropout_ratio = local_channel_dropout_ratio
        self.image_channel_dropout_ratio = image_channel_dropout_ratio
        self.salt_pepper_ratio = salt_pepper_ratio
        self.salt_pepper_scales = salt_pepper_scales
        self.data_format = data_format
        self.activation_fn, self.kernel_initializer, self.alpha_dropout = activation_fn_kernel_initializer_alpha_dropout(activation)
        self.output_activation_fn, self.output_layer_kernel_initializer = activation_fn_output_kernel_initializer(prediction_activation, self.data_format)

        if output_background_local:
            self.local_channel_dropout = Dropout(self.local_channel_dropout_ratio, noise_shape=[1, self.num_labels, 1, 1, 1] if self.data_format == 'channels_first' else [1, 1, 1, 1, self.num_labels])
        else:
            self.local_channel_dropout = Dropout(self.local_channel_dropout_ratio, noise_shape=[1, self.num_labels - 1, 1, 1, 1] if self.data_format == 'channels_first' else [1, 1, 1, 1, self.num_labels - 1])
        self.image_channel_dropout = Dropout(self.image_channel_dropout_ratio, noise_shape=[1, 1, 1, 1, 1] if self.data_format == 'channels_first' else [1, 1, 1, 1, 1])

        self.downsampling = AveragePooling3D([self.downsampling_factor] * 3, name='downsampling', data_format=self.data_format)
        self.unets = [Sequential([UnetAvgLinear3D(data_format=self.data_format, padding=self.padding, kernel_initializer=self.kernel_initializer, activation=self.activation_fn, alpha_dropout=self.alpha_dropout, name=f'unet_l{i}', **unet_parameters),
                                          Conv3D(1, [1] * 3, name=f'l{i}', kernel_initializer=self.output_layer_kernel_initializer, activation=None, data_format=self.data_format, padding=self.padding)])
                              for i in range(self.num_labels - (not self.output_background))]
        self.upsampling = UpSampling3DCubic([self.downsampling_factor] * 3, name='upsampling', data_format=self.data_format)

    def spatial_configuration_component(self, local_appearance, image, training):
        channel_axis = get_channel_index(local_appearance, self.data_format)
        local_appearance_split = tf.unstack(local_appearance, axis=channel_axis, name='split')
        image_split = tf.unstack(image, axis=channel_axis, name='split_img')
        if self.input_background:
            if len(local_appearance_split) != self.num_labels:
                raise RuntimeError('background label not present in input to SpatialConfiguration')
        else:
            if len(local_appearance_split) == self.num_labels:
                local_appearance_split = local_appearance_split[1:]
        spatial_configuration_split = []
        for i in range(self.num_labels):
            if not self.output_background and i == 0:
                continue
            if self.input_background:
                input_indizes = [j for j in range(0, self.num_labels) if j != i]
            else:
                input_indizes = [j - 1 for j in range(1, self.num_labels) if j != i]
            print(f'label {i} input {input_indizes}')

            if self.image_as_input:
                local_appearance_wo_l_i = tf.stack(image_split + [local_appearance_split[j] for j in input_indizes], axis=channel_axis, name=f'input_l{i}')
            else:
                local_appearance_wo_l_i = tf.stack([local_appearance_split[j] for j in input_indizes], axis=channel_axis, name=f'input_l{i}')

            spatial_configuration_l_i = self.unets[i - (not self.input_background)](local_appearance_wo_l_i, training=training)
            spatial_configuration_split.append(spatial_configuration_l_i)
        return tf.concat(spatial_configuration_split, channel_axis, name='merge')

    def call(self, local_prediction_and_image, training, **kwargs):
        local_prediction, image = local_prediction_and_image
        node = local_prediction
        node = self.downsampling(node, training=training)
        if self.salt_pepper_ratio > 0 and training:
            node = salt_pepper_3D(node, self.salt_pepper_ratio, scales=self.salt_pepper_scales, data_format=self.data_format)
        node = self.local_channel_dropout(node, training=training)
        if training:
            node = self.local_channel_dropout(node, training=training)

        node_img = image
        if self.image_as_input:
            node_img = self.downsampling(node_img, training=training)

            if self.image_channel_dropout_ratio > 0 and training:
                node_img = self.image_channel_dropout(node_img, training=training)

        node = self.spatial_configuration_component(node, node_img, training=training)
        spatial_prediction_wo_sigmoid = self.upsampling(node, training=training)
        return spatial_prediction_wo_sigmoid


