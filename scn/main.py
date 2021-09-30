#!/usr/bin/python
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import utils.io.image
from tensorflow_train_v2.losses.semantic_segmentation_losses import generalized_dice_loss, sigmoid_cross_entropy_with_logits
from tensorflow_train_v2.train_loop import MainLoopBase
import utils.sitk_image
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric, SurfaceDistanceMetric
from tensorflow_train_v2.dataset.dataset_iterator_multiprocessing import DatasetIteratorMultiprocessing as DatasetIterator
from dataset import Dataset
from network import LocalAppearance, SpatialConfiguration, Final


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

        # TODO set dataset and output folder
        self.base_dataset_folder = '/SET/PATH/TO/DATASET'
        self.base_output_folder = '/SET/PATH/TO/OUTPUT_FOLDER'

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

        self.final_network_parameters = dict(data_format=self.data_format,
                                             local_prediction_activation=self.network_parameters['local_activation'],
                                             spatial_prediction_activation=self.network_parameters['spatial_activation'])

        self.dataset_parameters = dict(base_folder=self.base_dataset_folder,
                                       image_spacing=list(reversed(self.image_spacing)),
                                       cv=self.training_parameters['cv'],
                                       data_format=self.data_format,
                                       image_pixel_type=np.float16 if mixed_precision else np.float32,
                                       use_landmarks=True,
                                       save_debug_images=False,
                                       **self.dataset_parameters)

        self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice']])

        self.load_model_filename = training_parameters.get('load_model_filename', None)
        if self.load_model_filename is not None:
            self.test_initialization = True
            self.load_model_iter = training_parameters['load_model_iter']

        if self.use_mixed_precision:
            dtype = tf.float16
        else:
            dtype = tf.float32
        self.call_model_and_loss = tf.function(self.call_model_and_loss, input_signature=[tf.TensorSpec([None, 1, None, None, None], dtype=dtype),
                                                                                          tf.TensorSpec([None, self.num_labels, None, None, None], dtype=tf.uint8),
                                                                                          tf.TensorSpec([], dtype=tf.bool)])

    def init_model(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.norm_moving_average = tf.Variable(10.0)
        self.local_appearance = LocalAppearance(**self.local_network_parameters)
        self.spatial_configuration = SpatialConfiguration(**self.spatial_network_parameters)
        self.final = Final(**self.final_network_parameters)
        self.model_list = [self.local_appearance,
                           self.spatial_configuration,
                           self.final,
                           ]
        self.call_model(tf.zeros([1, 1] + [64, 64, 64]), training=False)
        self.all_model_trainable_variables = [var for model in self.model_list for var in model.trainable_variables]

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
        self.checkpoint = tf.train.Checkpoint(local_appearance=self.local_appearance, spatial_configuration=self.spatial_configuration, final=self.final, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name='scn', additional_info=self.output_folder_name)

    def init_datasets(self):
        network_image_size = [None, None, None]

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('labels', [1] + network_image_size),
                                                  ])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('labels', network_image_size + [1]),
                                                  ])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32,
                                'labels': tf.uint8,
                                }

        cache_maxsize = 8192

        self.dataset_parameters['cache_maxsize'] = cache_maxsize
        dataset = Dataset(**self.dataset_parameters)
        self.dataset_train = dataset.dataset_train()
        self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size, n_threads=self.training_parameters['dataset_threads'], queue_size=16)
        self.dataset_val = dataset.dataset_val()

    def init_loggers(self):
        self.loss_metric_logger_train = LossMetricLogger('train',
                                                         self.output_folder_handler.path('train'),
                                                         self.output_folder_handler.path('train.csv'))
        self.loss_metric_logger_val = LossMetricLogger('test',
                                                       self.output_folder_handler.path('test'),
                                                       self.output_folder_handler.path('test.csv'))

    def split_labels_tf(self, labels, w_batch_dim):
        if w_batch_dim:
            axis = self.channel_axis
        else:
            axis = 0 if self.data_format == 'channels_first' else -1  # axis wo batch dimension
        split_labels = tf.one_hot(tf.squeeze(labels, axis=axis), depth=self.num_labels, axis=axis, dtype=labels.dtype)
        return split_labels

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

    # tf.function defined in __init__
    def call_model_and_loss(self, image, labels, training):
        prediction, prediction_wo_activation, local_prediction, spatial_prediction, local_prediction_wo_activation, spatial_prediction_wo_activation = self.call_model(image, training=training)
        losses = self.supervised_losses(labels, prediction, local_prediction_wo_activation, spatial_prediction_wo_activation, **self.loss_parameters)
        return (prediction, prediction_wo_activation, local_prediction, spatial_prediction, local_prediction_wo_activation, spatial_prediction_wo_activation), losses

    @tf.function
    def train_step(self):
        image, labels = self.dataset_train_iter.get_next()
        labels = self.split_labels_tf(labels, w_batch_dim=True)
        all_variables = self.all_model_trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(all_variables)

            _, losses = self.call_model_and_loss(image, labels, training=True)
            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.local_appearance.losses + self.spatial_configuration.losses + self.final.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        metric_dict = losses
        clip_norm = self.norm_moving_average * 5
        if self.use_mixed_precision:
            scaled_grads = tape.gradient(scaled_loss, all_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            loss_scale = self.optimizer.loss_scale
            metric_dict.update({'loss_scale': loss_scale})
        else:
            grads = tape.gradient(loss, all_variables)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)

        if tf.math.is_finite(norm):
            alpha = 0.01
            self.norm_moving_average.assign(alpha * tf.minimum(norm, clip_norm) + (1 - alpha) * self.norm_moving_average)
        metric_dict.update({'norm': norm, 'norm_average': self.norm_moving_average})
        self.optimizer.apply_gradients(zip(grads, all_variables))
        self.ema.apply(all_variables)
        self.loss_metric_logger_train.update_metrics(metric_dict)

    @tf.function
    def supervised_losses(self, labels, prediction, local_prediction_wo_activation, spatial_prediction_wo_activation, loss_factor=0, loss_factor_local=0, loss_factor_spatial=0, suffix='', **kwargs):
        losses_dict = {}
        if labels is not None:
            prediction = tf.cast(prediction, tf.float32)
            labels_wo_background = tf.stack(tf.unstack(labels, axis=self.channel_axis)[1:], axis=self.channel_axis)
            if loss_factor > 0:
                loss_final = generalized_dice_loss(labels=labels if self.output_background_final else labels_wo_background, logits_as_probability=prediction, data_format=self.data_format)
                losses_dict['loss' + suffix] = loss_factor * loss_final
            if loss_factor_local > 0:
                local_prediction_wo_activation = tf.cast(local_prediction_wo_activation, tf.float32)
                losses_dict['loss_local' + suffix] = loss_factor_local * sigmoid_cross_entropy_with_logits(labels=labels if self.output_background_local else labels_wo_background, logits=local_prediction_wo_activation, data_format=self.data_format)
            if loss_factor_spatial > 0:
                spatial_prediction_wo_activation = tf.cast(spatial_prediction_wo_activation, tf.float32)
                losses_dict['loss_spatial' + suffix] = loss_factor_spatial * sigmoid_cross_entropy_with_logits(labels=labels if self.output_background_spatial else labels_wo_background, logits=spatial_prediction_wo_activation, data_format=self.data_format)
        return losses_dict

    def get_summary_dict(self, segmentation_statistics, name):
        mean_list = segmentation_statistics.get_metric_mean_list(name)
        mean_of_mean_list = np.mean(mean_list)
        return OrderedDict(list(zip(self.metric_names[name], [mean_of_mean_list] + mean_list)))

    def get_model_prediction(self, dataset_entry):
        generators = dataset_entry['generators']
        if self.has_validation_groundtruth:
            (prediction, prediction_wo_activation, local_prediction, spatial_prediction, local_prediction_wo_activation, spatial_prediction_wo_activation), losses = self.call_model_and_loss(np.expand_dims(generators['image'], axis=0), np.expand_dims(generators['labels'], axis=0), False)
        else:
            prediction, prediction_wo_activation, local_prediction, spatial_prediction, local_prediction_wo_activation, spatial_prediction_wo_activation = self.call_model(np.expand_dims(generators['image'], axis=0), False)

        prediction = np.squeeze(prediction.numpy(), axis=0)
        local_prediction = np.squeeze(local_prediction.numpy(), axis=0)
        spatial_prediction = np.squeeze(spatial_prediction.numpy(), axis=0)
        prediction_numpy_dict = {
                                 'prediction': prediction,
                                 'local_prediction': local_prediction,
                                 'spatial_prediction': spatial_prediction,
                                 }
        return prediction_numpy_dict

    def process_prediction_and_calculate_metrics(self, prediction_numpy_dict, dataset_entry, segmentation_statistics, segmentation_test, metrics_dict):
        prediction = prediction_numpy_dict['prediction']
        current_id = dataset_entry['id']['image_id']
        datasources = dataset_entry['datasources']
        transformations = dataset_entry['transformations']
        input = datasources['image']
        transformation = transformations['image']
        prediction_labels = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation)
        utils.io.image.write(prediction_labels, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '.nii.gz'))

        if self.has_validation_groundtruth:
            groundtruth = datasources['labels']
            metric_values = segmentation_statistics.get_metric_values(prediction_labels, groundtruth)
            metrics_dict[current_id] = metric_values

    def test(self):
        if self.training_parameters['cv'] == 0:
            print(f'Skip Testing, cv=0')
            return

        print('Testing...')

        if self.current_iter != 0:
            old_weights = [tf.keras.backend.get_value(var) for var in self.all_model_trainable_variables]
            new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.all_model_trainable_variables]
            for var, weights in zip(self.all_model_trainable_variables, new_weights):
                tf.keras.backend.set_value(var, weights)

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        labels_wo_background = list(range(1, self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='linear',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)
        segmentation_statistics = SegmentationStatistics(labels_wo_background,
                                                         self.output_folder_handler.path_for_iteration(self.current_iter),
                                                         metrics=OrderedDict([('dice', DiceMetric())]))

        num_entries = self.dataset_val.num_entries()
        metrics_dict = {}
        for _ in tqdm(range(num_entries), desc=f'Testing'):
            dataset_entry = self.dataset_val.get_next()
            dataset_entry['generators']['labels'] = self.split_labels_tf(dataset_entry['generators']['labels'], w_batch_dim=False)
            prediction_numpy_dict = self.get_model_prediction(dataset_entry)

            self.process_prediction_and_calculate_metrics(prediction_numpy_dict, dataset_entry, segmentation_statistics, segmentation_test, metrics_dict)

        if self.has_validation_groundtruth:
            metrics_dict = dict(sorted(metrics_dict.items(), key=lambda item: item[0]))
            for key, value in metrics_dict.items():
                segmentation_statistics.set_metric_values(key, value)
            segmentation_statistics.finalize()

            summary_values = OrderedDict()
            for name in self.metric_names.keys():
                summary_values.update(self.get_summary_dict(segmentation_statistics, name))
            self.loss_metric_logger_val.update_metrics(summary_values)
            self.loss_metric_logger_val.finalize(self.current_iter)

        if self.current_iter != 0:
            for var, weights in zip(self.all_model_trainable_variables, old_weights):
                tf.keras.backend.set_value(var, weights)


def run():
    training_parameters = dict(
        learning_rate=0.0001,
        max_iter=100000,  # 500000
        test_iter=10000,  # 20000
        test_initialization=False,
        num_labels=5,
        image_spacing=[2] * 3,
        output_background_local=True,
        input_background_spatial=False,
        output_background_spatial=True,
        output_background_final=True,
        dataset_threads=8,
        load_model_filename=None,
    )

    dataset_parameters = dict(
        cached_datasource=False,
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
        stop_gradient_spatial=True)

    for i in [1, 2, 3, 4]:  # 1, 2, 3, 4: train respective cross-validation fold(s); 0: use all training data
        training_parameters['cv'] = i
        output_folder_name = f'segmentation/cv{i}/'
        loop = MainLoop(training_parameters=training_parameters,
                        dataset_parameters=dataset_parameters,
                        network_parameters=network_parameters,
                        loss_parameters=loss_parameters,
                        output_folder_name=output_folder_name)
        loop.run()


if __name__ == '__main__':
    run()



