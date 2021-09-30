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
from network import Unet, UnetAvgLinear3D
import itertools


class MainLoop(MainLoopBase):
    def __init__(self,
                 cv,
                 network,
                 unet,
                 normalized_prediction,
                 loss,
                 network_parameters,
                 learning_rate,
                 cached_datasource=True,
                 dataset_threads=4,
                 output_folder_name='',
                 load_model_filename=None):
        super().__init__()

        self.use_mixed_precision = True
        if self.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.cv = cv
        self.batch_size = 1
        self.learning_rate = learning_rate
        self.max_iter = 100000  # 1000000
        self.test_iter = 10000  # 50000
        self.disp_iter = 10
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.000001
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.normalized_prediction = normalized_prediction
        self.padding = 'same'
        self.output_folder_name = output_folder_name

        self.dataset_threads = dataset_threads
        self.has_validation_groundtruth = cv != 0

        # TODO set dataset and output folder
        self.base_dataset_folder = '/SET/PATH/TO/DATASET'
        self.base_output_folder = '/SET/PATH/TO/OUTPUT_FOLDER'

        self.num_labels = 2
        self.merge_foreground_labels = True

        self.image_spacing = [6] * 3
        self.valid_output_sizes = [[32, 48, 64, 80], [32, 48, 64, 80], list(range(32, 257, 16))]
        self.min_output_sizes = [min(x) for x in self.valid_output_sizes]

        self.network_parameters = dict(num_labels=self.num_labels,
                                       actual_network=self.unet,
                                       padding=self.padding,
                                       data_format=self.data_format,
                                       **network_parameters)

        self.dataset_parameters = dict(base_folder=self.base_dataset_folder,
                                       valid_output_sizes=self.valid_output_sizes,
                                       image_spacing=list(reversed(self.image_spacing)),
                                       cv=self.cv,
                                       num_labels=self.num_labels,
                                       merge_foreground_labels=self.merge_foreground_labels,
                                       cached_datasource=cached_datasource,
                                       data_format=self.data_format,
                                       image_pixel_type=np.float16 if mixed_precision else np.float32,
                                       input_gaussian_sigma=3.0,
                                       label_gaussian_sigma=0.0,
                                       use_landmarks=False,
                                       save_debug_images=False)

        self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice']])
        self.loss_function = loss

        self.training_parameters = {
                                    'cv': self.cv,
                                    'batch_size': self.batch_size,
                                    'learning_rate': self.learning_rate,
                                    'max_iter': self.max_iter,
                                    'test_iter': self.test_iter,
                                    'disp_iter': self.disp_iter,
                                    'snapshot_iter': self.snapshot_iter,
                                    'test_initialization': self.test_initialization,
                                    'current_iter': self.current_iter,
                                    'reg_constant': self.reg_constant,
                                    'data_format': self.data_format,
                                    'channel_axis': self.channel_axis,
                                    'network': self.network,
                                    'unet': self.unet,
                                    'normalized_prediction': self.normalized_prediction,
                                    'padding': self.padding,
                                    'output_folder_name': self.output_folder_name,
                                    'dataset_threads': self.dataset_threads,
                                    'has_validation_groundtruth': self.has_validation_groundtruth,
                                    'num_labels': self.num_labels,
                                    'image_spacing': self.image_spacing,
                                    'loss_function': self.loss_function,
                                    }

        self.load_model_filename = load_model_filename
        if self.load_model_filename is not None:
            self.test_initialization = True

        if self.use_mixed_precision:
            dtype = tf.float16
        else:
            dtype = tf.float32
        self.call_model_and_loss = tf.function(self.call_model_and_loss, input_signature=[tf.TensorSpec([None, 1, None, None, None], dtype=dtype),
                                                                                          tf.TensorSpec([None, self.num_labels, None, None, None], dtype=tf.uint8),
                                                                                          tf.TensorSpec([], dtype=tf.bool),
                                                                                          ])

    def init_model(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.norm_moving_average = tf.Variable(10.0)
        self.model = self.network(**self.network_parameters)

    def save_model(self):
        """
        Save the model.
        """
        old_weights = [tf.keras.backend.get_value(var) for var in self.model.trainable_variables]
        new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.model.trainable_variables]
        for var, weights in zip(self.model.trainable_variables, new_weights):
            tf.keras.backend.set_value(var, weights)
        super(MainLoop, self).save_model()
        for var, weights in zip(self.model.trainable_variables, old_weights):
            tf.keras.backend.set_value(var, weights)

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=1, increment_period=100))

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name, additional_info=self.output_folder_name)

    def init_datasets(self):
        network_image_size = [None, None, None]

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('labels', [self.num_labels] + network_image_size),
                                                  ])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('labels', network_image_size + [self.num_labels]),
                                                  ])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32,
                                'labels': tf.uint8,
                                }

        cache_maxsize = 8192
        self.dataset_parameters['cache_maxsize'] = cache_maxsize
        dataset = Dataset(**self.dataset_parameters)
        self.dataset_train = dataset.dataset_train()
        self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size, n_threads=self.dataset_threads, queue_size=16)
        self.dataset_val = dataset.dataset_val()

    def init_loggers(self):
        self.loss_metric_logger_train = LossMetricLogger('train',
                                                         self.output_folder_handler.path('train'),
                                                         self.output_folder_handler.path('train.csv'))
        self.loss_metric_logger_val = LossMetricLogger('test',
                                                       self.output_folder_handler.path('test'),
                                                       self.output_folder_handler.path('test.csv'))

    # tf.function defined with input signature in __init__
    def call_model_and_loss(self, image, groundtruth, training):
        prediction = self.model(image, training=training)
        losses = self.losses(groundtruth, prediction)
        return prediction, losses

    @tf.function
    def train_step(self):
        image, label = self.dataset_train_iter.get_next()
        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(image, label, training=True)
            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        variables = self.model.trainable_weights
        metric_dict = losses
        clip_norm = self.norm_moving_average * 5
        if self.use_mixed_precision:
            scaled_grads = tape.gradient(scaled_loss, variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            loss_scale = self.optimizer.loss_scale
            metric_dict.update({'loss_scale': loss_scale})
        else:
            grads = tape.gradient(loss, variables)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
        if tf.math.is_finite(norm):
            alpha = 0.01
            self.norm_moving_average.assign(alpha * tf.minimum(norm, clip_norm) + (1 - alpha) * self.norm_moving_average)
        metric_dict.update({'norm': norm, 'norm_average': self.norm_moving_average})
        self.optimizer.apply_gradients(zip(grads, variables))
        self.ema.apply(variables)

        self.loss_metric_logger_train.update_metrics(metric_dict)

    @tf.function
    def losses(self, groundtruth, prediction):
        groundtruth = tf.cast(groundtruth, dtype=prediction.dtype)
        loss_total = self.loss_function(labels=groundtruth, logits=prediction, data_format=self.data_format)
        losses_dict = {'loss': loss_total}
        return losses_dict

    def get_summary_dict(self, segmentation_statistics, name):
        mean_list = segmentation_statistics.get_metric_mean_list(name)
        mean_of_mean_list = np.mean(mean_list)
        return OrderedDict(list(zip(self.metric_names[name], [mean_of_mean_list] + mean_list)))

    def get_model_prediction(self, dataset_entry):
        generators = dataset_entry['generators']
        if self.has_validation_groundtruth:
            (prediction), losses = self.call_model_and_loss(np.expand_dims(generators['image'], axis=0), np.expand_dims(generators['labels'], axis=0), False)
        else:
            prediction = self.model(np.expand_dims(generators['image'], axis=0), False)

        prediction_np = np.squeeze(prediction, axis=0)
        prediction_w_activation_np = np.squeeze(tf.nn.softmax(prediction, axis=1 if self.data_format == 'channels_first' else -1), axis=0)

        prediction_numpy_dict = {
            'prediction': prediction_np,
            'prediction_w_activation': prediction_w_activation_np,
        }
        return prediction_numpy_dict

    def process_prediction_and_calculate_metrics(self, prediction_numpy_dict, dataset_entry, segmentation_statistics, segmentation_test, landmarks_dict, metrics_dict):
        prediction = prediction_numpy_dict['prediction']
        prediction_w_activation = prediction_numpy_dict['prediction_w_activation']

        current_id = dataset_entry['id']['image_id']
        datasources = dataset_entry['datasources']
        transformations = dataset_entry['transformations']

        input_image = datasources['image']
        transformation = transformations['image']

        prediction_labels = segmentation_test.get_label_image(prediction, input_image, self.image_spacing, transformation)
        utils.io.image.write(prediction_labels, self.output_folder_handler.path(current_id + '.mha'))

        if self.has_validation_groundtruth:
            groundtruth = datasources['labels']
            metric_values = segmentation_statistics.get_metric_values(prediction_labels, groundtruth)
            metrics_dict[current_id] = metric_values

        def bbox_ND(img):
            N = img.ndim
            bb_min = []
            bb_max = []
            for ax in itertools.combinations(reversed(range(N)), N - 1):
                nonzero = np.any(img, axis=ax)
                min_val, max_val = np.where(nonzero)[0][[0, -1]]
                bb_min.append(int(min_val))
                bb_max.append(int(max_val))
            return bb_min, bb_max

        def correct_min_max_order(min_point, max_point):
            for i, (min_v, max_v) in enumerate(zip(min_point, max_point)):
                if min_v > max_v:
                    min_point[i] = max_v
                    max_point[i] = min_v
            return min_point, max_point

        labels_np = np.argmax(prediction_w_activation, axis=0)
        labels_np = np.transpose(labels_np, axes=[2, 1, 0])
        bb_min, bb_max = bbox_ND(labels_np)

        def transform_points(point, spacing, transformation):
            tmp = [p * s for p, s in zip(point, spacing)]
            point_transformed = transformation.TransformPoint(tmp)
            return point_transformed

        bb_min_physical = list(transform_points(bb_min, self.image_spacing, transformation))
        bb_max_physical = list(transform_points(bb_max, self.image_spacing, transformation))
        bb_min_physical, bb_max_physical = correct_min_max_order(bb_min_physical, bb_max_physical)

        curr_landmarks = []
        curr_landmarks.append(utils.landmark.common.Landmark(bb_min_physical))
        curr_landmarks.append(utils.landmark.common.Landmark(bb_max_physical))

        landmarks_dict[current_id] = curr_landmarks

    def test(self):

        if self.training_parameters['cv'] == 0:
            print(f'Skip Testing, cv=0')
            return

        print('Testing...')
        self.output_folder_handler.set_current_path_for_iteration(self.current_iter)

        if not self.first_iteration:
            old_weights = [tf.keras.backend.get_value(var) for var in self.model.trainable_variables]
            new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.model.trainable_variables]
            for var, weights in zip(self.model.trainable_variables, new_weights):
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
                                                         self.output_folder_handler.path(),
                                                         metrics=OrderedDict([('dice', DiceMetric()),
                                                                              ]))

        landmarks_dict = {}

        num_entries = self.dataset_val.num_entries()
        metrics_dict = {}
        for _ in tqdm(range(num_entries), desc=f'Testing'):
            dataset_entry = self.dataset_val.get_next()
            prediction_numpy_dict = self.get_model_prediction(dataset_entry)

            self.process_prediction_and_calculate_metrics(prediction_numpy_dict, dataset_entry, segmentation_statistics, segmentation_test, landmarks_dict, metrics_dict)

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

        utils.io.landmark.save_points_csv(landmarks_dict,
                                          self.output_folder_handler.path('landmark_bb.csv'))

        if not self.first_iteration:
            for var, weights in zip(self.model.trainable_variables, old_weights):
                tf.keras.backend.set_value(var, weights)


def run():
    network_parameters = {'num_filters_base': 32,
                          'num_levels': 5,
                          'dropout_ratio': 0.1,
                          'activation': 'lrelu',
                          }

    cached_datasource = False
    dataset_threads = 8

    for i in [1, 2, 3, 4]:  # 1, 2, 3, 4: train respective cross-validation fold(s); 0: use all training data
        output_folder_name = f'localization/cv{i}/'
        loop = MainLoop(cv=i,
                        network=Unet,
                        unet=UnetAvgLinear3D,
                        normalized_prediction=False,
                        loss=generalized_dice_loss,
                        network_parameters=network_parameters,
                        learning_rate=1e-5,
                        cached_datasource=cached_datasource,
                        dataset_threads=dataset_threads,
                        output_folder_name=output_folder_name)
        loop.run()


if __name__ == '__main__':
    run()
