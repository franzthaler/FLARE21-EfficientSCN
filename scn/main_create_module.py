#!/usr/bin/python
import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from network_inference import LocalAppearance, SpatialConfiguration


class MainLoop(MainLoopBase):
    def __init__(self,
                 training_parameters,
                 dataset_parameters,
                 network_parameters,
                 loss_parameters,
                 output_folder_name=''):

        super().__init__()

        self.use_mixed_precision = True
        if self.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.training_parameters = training_parameters
        self.dataset_parameters = dataset_parameters
        self.network_parameters = network_parameters
        self.loss_parameters = loss_parameters

        self.batch_size = 1
        self.learning_rate = self.training_parameters['learning_rate']
        self.max_iter = self.training_parameters['max_iter']
        self.test_iter = self.training_parameters['test_iter']
        self.disp_iter = 10
        self.snapshot_iter = self.test_iter
        self.test_initialization = self.training_parameters['test_initialization']
        self.current_iter = 0
        self.reg_constant = 0.000001
        self.data_format = 'channels_first'
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.padding = 'same'
        self.output_folder_name = output_folder_name

        self.num_labels = training_parameters['num_labels']
        self.image_spacing = training_parameters['image_spacing']

        self.output_background_local = self.training_parameters['output_background_local']
        self.input_background_spatial = self.training_parameters['input_background_spatial']
        self.output_background_spatial = self.training_parameters['output_background_spatial']
        self.output_background_final = self.training_parameters['output_background_final']

        self.has_validation_groundtruth = self.training_parameters['cv'] != 0

        # FLARE21
        self.local_base_folder = '/media0/franz/datasets/multi_organ/flare21'
        self.base_output_folder = '/media0/franz/experiments/semi_supervised_learning/flare21'

        self.local_network_parameters = dict(num_labels=self.num_labels,
                                             padding=self.padding,
                                             data_format=self.data_format,
                                             activation=self.network_parameters['activation'],
                                             output_background=self.output_background_local,
                                             prediction_activation=self.network_parameters['local_activation'],
                                             unet_parameters=self.network_parameters['local_network_parameters'])

        self.spatial_network_parameters = dict(num_labels=self.num_labels,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               activation=self.network_parameters['activation'],
                                               input_background=self.input_background_spatial,
                                               output_background=self.output_background_spatial,
                                               downsampling_factor=self.network_parameters['spatial_downsample'],
                                               local_channel_dropout_ratio=self.network_parameters['local_channel_dropout_ratio'],
                                               image_channel_dropout_ratio=self.network_parameters['image_channel_dropout_ratio'],
                                               prediction_activation=self.network_parameters['spatial_activation'],
                                               salt_pepper_ratio=self.network_parameters['spatial_salt_pepper_ratio'],
                                               salt_pepper_scales=self.network_parameters['spatial_salt_pepper_scales'],
                                               output_background_local=self.training_parameters['output_background_local'],
                                               image_as_input=self.network_parameters['spatial_network_image_as_input'],
                                               unet_parameters=self.network_parameters['spatial_network_parameters'])

        self.load_model_filename = training_parameters.get('load_model_filename', None)
        if self.load_model_filename is not None:
            self.test_initialization = True
            self.load_model_iter = training_parameters['load_model_iter']

    def init_model(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.norm_moving_average = tf.Variable(10.0)
        self.local_appearance = LocalAppearance(**self.local_network_parameters)
        self.spatial_configuration = SpatialConfiguration(**self.spatial_network_parameters)
        self.model_list = [self.local_appearance,
                           self.spatial_configuration,
                           ]
        self.all_model_trainable_variables = [var for model in self.model_list for var in model.trainable_variables]

    def load_model(self, model_filename=None, assert_consumed=False):
        """
        Load the model.
        """
        model_filename = model_filename or self.load_model_filename
        print('Restoring model ' + model_filename)
        status = self.checkpoint.restore(model_filename)
        if assert_consumed:
            status.assert_consumed()
        else:
            status.expect_partial()

    def save_model(self):
        """
        Save the model.
        """
        old_weights = [tf.keras.backend.get_value(var) for var in self.all_model_trainable_variables]
        new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.all_model_trainable_variables]
        for var, weights in zip(self.all_model_trainable_variables, new_weights):
            tf.keras.backend.set_value(var, weights)
        super(MainLoop, self).save_model()
        for var, weights in zip(self.all_model_trainable_variables, old_weights):
            tf.keras.backend.set_value(var, weights)

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=2 ** 15, increment_period=1000))

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(local_appearance=self.local_appearance, spatial_configuration=self.spatial_configuration)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name='scn', additional_info=self.output_folder_name)

    @tf.function
    def call_model(self, image, training):
        local_prediction, local_prediction_wo_activation = self.local_appearance(image, training=training)
        spatial_network_input = local_prediction

        if self.network_parameters['stop_gradient_spatial']:
            spatial_network_input = tf.stop_gradient(spatial_network_input)
        spatial_prediction, spatial_prediction_wo_activation = self.spatial_configuration([spatial_network_input, image], training=training)

        final_network_input_local = tf.cast(local_prediction_wo_activation, tf.float32)
        final_network_input_spatial = tf.cast(spatial_prediction_wo_activation, tf.float32)

        prediction, prediction_wo_activation = self.final([final_network_input_local, final_network_input_spatial], training=training)
        return prediction, prediction_wo_activation, local_prediction, spatial_prediction, local_prediction_wo_activation, spatial_prediction_wo_activation


class CombinedModule(tf.Module):
    def __init__(self, local_appearance, spatial_configuration, data_format='channels_first'):
        super(CombinedModule, self).__init__()
        self.local_appearance = local_appearance
        self.spatial_configuration = spatial_configuration
        self.data_format = data_format
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.prediction_fn = lambda x: x / (tf.reduce_sum(x, axis=1 if self.data_format == 'channels_first' else -1) + 1e-5)


    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16)])  # mixed precision
    # @tf.function(input_signature=[tf.TensorSpec([None, None, None, None, 1], tf.float16)])  # mixed precision
    def __call__(self, image, training=False):
        local_prediction, local_prediction_wo_activation = self.local_appearance(image, training=training)
        spatial_network_input = local_prediction
        spatial_prediction_wo_activation = self.spatial_configuration([spatial_network_input, image], training=training)
        prediction_w_activation = self.final(local_prediction_wo_activation, spatial_prediction_wo_activation, training=training)
        return prediction_w_activation

    def final(self, local_prediction, spatial_prediction, training=False):
        prediction_wo_activation = self.merge_based_on_spatial_uncertainty(local_prediction, spatial_prediction)
        prediction = self.prediction_fn(prediction_wo_activation)
        return prediction

    def merge_based_on_spatial_uncertainty(self, local_wo_activation, spatial_wo_activation):
        local = tf.nn.sigmoid(local_wo_activation)
        spatial = tf.nn.sigmoid(spatial_wo_activation)

        def sigmoid_uncertainty(tensor_wo_sigmoid, exponent=1.0):  # Note: data_format needed to use alias for either sigmoid or softmax
            cross_entropy_max = 0.6931471805599453  # -(0.5 * tf.math.log(0.5)) * 2
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(tensor_wo_sigmoid), logits=tensor_wo_sigmoid)
            cross_entropy_normalized = cross_entropy / cross_entropy_max
            cross_entropy_normalized = 1 - tf.pow(1.0 - cross_entropy_normalized, exponent)
            return cross_entropy_normalized

        spatial_uncertainty = sigmoid_uncertainty(spatial_wo_activation)
        return local * spatial_uncertainty + spatial * (1 - spatial_uncertainty)


def run():
    training_parameters = dict(
                               learning_rate=0.0001,
                               max_iter=200000,
                               test_iter=10000,
                               test_initialization=False,
                               num_labels=5,
                               image_size=[128, 128, 192],
                               image_spacing=[2] * 3,
                               output_background_local=True,
                               input_background_spatial=False,
                               output_background_spatial=True,
                               output_background_final=True,
                               dataset_threads=8,
                               load_model_filename=None,
                               )

    dataset_parameters = dict(
        setup_folder_to_use='setup',
        cached_datasource=True,
        input_gaussian_sigma=1.0,
        label_gaussian_sigma=1.0,
    )

    loss_parameters = dict(
        loss_factor=1.0,
        loss_factor_local=1.0,
        loss_factor_spatial=1.0,
    )

    network_parameters = dict(
                              local_network_parameters={'num_filters_base': 32, 'num_levels': 5, 'dropout_ratio': 0.1},
                              spatial_network_parameters={'num_filters_base': 16, 'num_levels': 4, 'dropout_ratio': 0.1},
                              spatial_downsample=4,
                              activation='lrelu',
                              local_channel_dropout_ratio=0.25,
                              image_channel_dropout_ratio=0.5,
                              local_activation='sigmoid',
                              spatial_activation='sigmoid',
                              spatial_salt_pepper_ratio=0.01,
                              spatial_salt_pepper_scales=[1, 2, 4, 8],
                              spatial_network_image_as_input=True,
                              stop_gradient_spatial=True,
                              stop_gradient_final=False,
                              )

    # TODO set model path and iter to load
    load_model_base = '/SET/PATH/TO/A/FINISHED/EXPERIMENT'
    load_model_iter = 100000  # set this to the final or an intermediate number of iterations for which model weights where saved

    training_parameters['load_model_filename'] = os.path.join(load_model_base, 'weights', f'ckpt-{load_model_iter}')
    training_parameters['load_model_iter'] = load_model_iter

    training_parameters['cv'] = 0
    output_folder_name = f'segmentation_create_module/'

    loop = MainLoop(training_parameters=training_parameters,
                    dataset_parameters=dataset_parameters,
                    network_parameters=network_parameters,
                    loss_parameters=loss_parameters,
                    output_folder_name=output_folder_name)

    loop.init_model()
    loop.init_optimizer()
    loop.init_output_folder_handler()
    loop.init_checkpoint()
    loop.init_checkpoint_manager()
    if loop.load_model_filename is not None:
        loop.load_model()

    combined_module_folder = '../saved_models/segmentation'
    combined_module = CombinedModule(loop.local_appearance, loop.spatial_configuration)

    data_format = loop.data_format
    if data_format == 'channels_first':
        image = tf.zeros([1, 1, 64, 64, 64], dtype=tf.float16)
    else:
        image = tf.zeros([1, 64, 64, 64, 1], dtype=tf.float16)
    prediction = combined_module(image)  # forward pass to trace model such that it can be saved

    tf.saved_model.save(combined_module, combined_module_folder)


if __name__ == '__main__':
    run()



