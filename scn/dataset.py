import os

import numpy as np
import SimpleITK as sitk
from datasets.graph_dataset import GraphDataset
from random import choice

from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.image_size_generator import ImageSizeGenerator
from graph.node import LambdaNode
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from utils.random import bool_bernoulli
from transformations.spatial.displacement_field import DisplacementField

class Dataset(object):
    """
    The dataset that processes files from the FLARE21 challenge.
    """
    def __init__(self,
                 image_spacing,
                 base_folder,
                 cv,
                 cached_datasource,
                 cache_maxsize=8192,
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=1.0,
                 use_landmarks=True,
                 num_labels=5,
                 displacement_field_sampling_factor=2,
                 image_folder=None,
                 setup_folder=None,
                 image_filename_postfix='_0000',
                 image_filename_extension='.nii.gz',
                 labels_filename_postfix='',
                 labels_filename_extension='.nii.gz',
                 landmarks_file=None,
                 valid_output_sizes_x=None,
                 valid_output_sizes_y=None,
                 valid_output_sizes_z=None,
                 data_format='channels_first',
                 image_pixel_type=np.float32,
                 save_debug_images=False):
        """
        Initializer.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3). Or 0 if full training/testing.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param use_landmarks: If True, center on loaded landmarks, otherwise use image center.
        :param num_labels: The number of output labels.
        :param image_folder: If set, use this folder for loading the images, otherwise use FLARE21 default.
        :param setup_folder: If set, use this folder for loading the setup files, otherwise use FLARE21 default.
        :param image_filename_postfix: The image filename postfix.
        :param image_filename_extension: The image filename extension.
        :param labels_filename_postfix: The labels filename postfix.
        :param labels_filename_extension: The labels filename extension.
        :param landmarks_file: If set, use this file for loading image landmarks, otherwise us FLARE21 default.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.cached_datasource = cached_datasource
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.use_landmarks = use_landmarks
        self.num_labels = num_labels
        self.displacement_field_sampling_factor = displacement_field_sampling_factor
        self.image_filename_postfix = image_filename_postfix
        self.image_filename_extension = image_filename_extension
        self.labels_filename_postfix = labels_filename_postfix
        self.labels_filename_extension = labels_filename_extension
        self.data_format = data_format
        self.image_pixel_type = image_pixel_type
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.cache_maxsize = cache_maxsize

        self.image_base_folder = image_folder or os.path.join(self.base_folder, 'TrainingImg')
        self.label_base_folder = image_folder or os.path.join(self.base_folder, 'TrainingMask')
        self.setup_base_folder = setup_folder or os.path.join(self.base_folder, 'setup')

        self.valid_output_sizes_x = valid_output_sizes_x or [32, 64, 96, 128, 160]
        self.valid_output_sizes_y = valid_output_sizes_y or [32, 64, 96, 128]
        self.valid_output_sizes_z = valid_output_sizes_z or [32, 64, 96, 128, 160]

        self.additional_extent = np.array([32, 32, 32])
        self.crop_randomly_smaller = False
        self.use_scale_if_bb_too_large = True

        self.landmarks_file = landmarks_file or os.path.join(self.setup_base_folder, 'landmark_bb.csv')
        self.num_landmarks = 2

        self.background_pixel_value = -1024

        self.postprocessing_random = self.intensity_postprocessing_ct_random
        self.postprocessing = self.intensity_postprocessing_ct

        if cv > 0:
            self.cv_folder = os.path.join(self.setup_base_folder, 'cv', str(cv))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'test.txt')
        else:
            self.train_file = os.path.join(self.setup_base_folder, 'cv', 'all_train.txt')
            self.test_file = os.path.join(self.setup_base_folder, 'cv', 'all_validation.txt')

    def datasources(self, iterator, cached, with_label=True):
        """
        Returns the data sources that load data.
        {
        'image:' (Cached)ImageDataSource that loads the image files.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        'mask:' (Cached)ImageDataSource that loads the groundtruth labels.
        }
        :param iterator: The iterator.
        :param cached: If True, use CachedImageDataSource.
        :return: A dict of data sources.
        """
        datasource_dict = {}
        if self.input_gaussian_sigma > 0.0:
            preprocessing = lambda image: gaussian_sitk(image, self.input_gaussian_sigma)
        else:
            preprocessing = None

        image_datasource_params = dict(root_location=self.image_base_folder,
                                       file_prefix='',
                                       file_suffix=self.image_filename_postfix,
                                       file_ext=self.image_filename_extension,
                                       set_zero_origin=False,
                                       set_identity_spacing=False,
                                       set_identity_direction=False,
                                       sitk_pixel_type=sitk.sitkInt16,
                                       preprocessing=preprocessing,
                                       name='image',
                                       parents=[iterator])
        if cached:
            image_datasource = CachedImageDataSource(cache_maxsize=self.cache_maxsize,
                                                     **image_datasource_params)
        else:
            image_datasource = ImageDataSource(**image_datasource_params)
        datasource_dict['image'] = image_datasource

        if self.use_landmarks:
            landmark_datasource = LandmarkDataSource(self.landmarks_file, self.num_landmarks, self.dim, name='landmarks', parents=[iterator])
            datasource_dict['landmarks'] = landmark_datasource
            datasource_dict['landmarks_bb'] = LambdaNode(self.image_landmark_bounding_box, name='landmarks_bb', parents=[datasource_dict['image'], datasource_dict['landmarks']])
            datasource_dict['landmarks_bb_start'] = LambdaNode(lambda x: x[0], name='landmarks_bb_start', parents=[datasource_dict['landmarks_bb']])
            datasource_dict['landmarks_bb_extent'] = LambdaNode(lambda x: x[1], name='landmarks_bb_extent', parents=[datasource_dict['landmarks_bb']])

        if with_label:
            mask_datasource_params = dict(root_location=self.label_base_folder,
                                          file_prefix='',
                                          file_suffix=self.labels_filename_postfix,
                                          file_ext=self.labels_filename_extension,
                                          set_zero_origin=False,
                                          set_identity_spacing=False,
                                          set_identity_direction=False,
                                          sitk_pixel_type=sitk.sitkUInt8,
                                          name='labels',
                                          parents=[iterator])
            if cached:
                mask_datasource = CachedImageDataSource(cache_maxsize=self.cache_maxsize,
                                                        **mask_datasource_params)
            else:
                mask_datasource = ImageDataSource(**mask_datasource_params)
            datasource_dict['labels'] = mask_datasource
        return datasource_dict

    def data_generators(self, datasources, transformation, image_post_processing, mask_post_processing, image_size, with_label=True, key_suffix=''):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: The datasources dictionary (see self.datasources()).
        :param transformation: The spatial transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :param image_size: The image size node.
        :return: A dict of data generators.
        """
        generator_dict = {}
        kwparents = {'output_size': image_size}
        image_generator = ImageGenerator(self.dim, image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format, name=f'image{key_suffix}', parents=[datasources['image'], transformation], resample_default_pixel_value=self.background_pixel_value, kwparents=kwparents)
        generator_dict[f'data{key_suffix}'] = image_generator
        if with_label:
            mask_image_generator = ImageGenerator(self.dim, image_size, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, data_format=self.data_format, name=f'labels{key_suffix}', parents=[datasources['labels'], transformation], kwparents=kwparents)
            generator_dict[f'labels{key_suffix}'] = mask_image_generator
        return generator_dict

    def smooth_labels(self, image):
        """
        Apply label smooth to a groundtruth label image.
        :param image: The groundtruth label image.
        :return: The smoothed label image.
        """
        if self.label_gaussian_sigma == 0.0:
            return image
        else:
            split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
            split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
            smoothed = np.stack(split_smoothed, 0)
            image_smoothed = np.expand_dims(np.argmax(smoothed, axis=0).astype(np.uint8), axis=0)
            return image_smoothed

    def set_spacing_if_1x1x1(self, image):
        spacing = list(image.GetSpacing())
        if all(x == 1 for x in spacing):
            # mean spacing of FLARE21 dataset
            mean = [0.814528640267884, 0.814528640267884, 2.52795318268203]
            image.SetSpacing(mean)
        return image

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               random_shift=0.2,
                               random_scale=0.2,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image).astype(self.image_pixel_type)

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image).astype(self.image_pixel_type)

    def spatial_transformation_augmented(self, datasources, image_size):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :param image_size: The image size node.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.use_landmarks:
            kwparents['start'] = datasources['landmarks_bb_start']
            kwparents['extent'] = datasources['landmarks_bb_extent']
            transformation_list.append(translation.BoundingBoxCenterToOrigin(self.dim, None, self.image_spacing))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    ])

        if self.use_scale_if_bb_too_large and self.use_landmarks:
            scale_factor_for_bb = LambdaNode(lambda output_size, start, extent, **kwargs: self.calculate_scale_from_bb(output_size, start, extent), kwparents={'output_size': image_size, 'start': datasources['landmarks_bb_start'], 'extent': datasources['landmarks_bb_extent']})
            transformation_list.append(LambdaNode(lambda s, **kwargs: scale.Fixed(self.dim, s, name='scale_if_bb_too_large').get(), kwparents={'s': scale_factor_for_bb}))

        transformation_list.extend([translation.OriginToOutputCenter(self.dim, None, self.image_spacing),
                                    deformation.Output(self.dim, [8, 8, 8], 15, None, self.image_spacing, spline_order=2),
                                    ])

        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return DisplacementField(output_size=None, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, name='image', parents=[comp], kwparents=kwparents)

    def spatial_transformation(self, datasources, image_size):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :param image_size: The image size node.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.use_landmarks:
            kwparents['start'] = datasources['landmarks_bb_start']
            kwparents['extent'] = datasources['landmarks_bb_extent']
            transformation_list.append(translation.BoundingBoxCenterToOrigin(self.dim, None, self.image_spacing))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))

        if self.use_scale_if_bb_too_large and self.use_landmarks:
            scale_factor_for_bb = LambdaNode(lambda output_size, start, extent, **kwargs: self.calculate_scale_from_bb(output_size, start, extent), kwparents={'output_size': image_size, 'start': datasources['landmarks_bb_start'], 'extent': datasources['landmarks_bb_extent']})
            transformation_list.append(LambdaNode(lambda s, **kwargs: scale.Fixed(self.dim, s, name='scale_if_bb_too_large').get(), kwparents={'s': scale_factor_for_bb}))

        transformation_list.append(translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def calculate_scale_from_bb(self, output_size, start, extent):
        expected_size = [int(x // s) for x, s in zip(extent, self.image_spacing)]
        scale_factor_x = expected_size[0] / max(self.valid_output_sizes_x)
        scale_factor_y = expected_size[1] / max(self.valid_output_sizes_y)
        scale_factor_z = expected_size[2] / max(self.valid_output_sizes_z)
        scale_factor_max = max(max(scale_factor_x, scale_factor_y), scale_factor_z)
        scale_factor = max(1, scale_factor_max)
        return [scale_factor] * self.dim

    def image_landmark_bounding_box(self, image, landmarks):
        """
        Calculate the bounding box from an image and landmarks.
        :param image: The image.
        :param landmarks: The landmarks.
        :return: (image, extent) tuple
        """
        all_coords = [l.coords for l in landmarks if l.is_valid]
        min_coords = np.min(all_coords, axis=0) - self.additional_extent
        max_coords = np.max(all_coords, axis=0) + self.additional_extent
        extent = max_coords - min_coords
        return min_coords, extent

    def image_bounding_box(self, image, bb):
        """
        Calculate the bounding box from an image and another bounding box.
        :param image: The image.
        :param bb: The bounding box.
        :return: (image, extent) tuple
        """
        all_coords = [np.array(list(map(float, bb[:3]))), np.array(list(map(float, bb[3:])))]
        image_min = np.array(image.GetOrigin())
        image_max = np.array([image.GetOrigin()[i] + image.GetSize()[i] * image.GetSpacing()[i] for i in range(3)])
        min_coords = np.min(all_coords, axis=0) - self.additional_extent
        min_coords = np.max([image_min, min_coords], axis=0)
        max_coords = np.max(all_coords, axis=0) + self.additional_extent
        max_coords = np.min([image_max, max_coords], axis=0)
        extent = max_coords - min_coords
        return min_coords, extent

    def crop_randomly_smaller_image_size(self, image_size):
        """
        Randomly use a smaller image size for a given image size.
        :param image_size: The image size.
        :return: The image size.
        """
        if bool_bernoulli(0.5):
            if image_size[2] == min(self.valid_output_sizes_z):
                return image_size
            smaller_sizes = [s for s in self.valid_output_sizes_z if s < image_size[2]]
            return image_size[:2] + [choice(smaller_sizes)]
        else:
            return image_size

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=self.cached_datasource)
        image_size = ImageSizeGenerator(self.dim, [None] * 3, self.image_spacing, valid_output_sizes=[self.valid_output_sizes_x, self.valid_output_sizes_y, self.valid_output_sizes_z], name='output_size', kwparents={'extent': sources['landmarks_bb_extent']})
        if self.crop_randomly_smaller:
            image_size = LambdaNode(self.crop_randomly_smaller_image_size, name='output_size', parents=[image_size])
        reference_transformation = self.spatial_transformation_augmented(sources, image_size)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.smooth_labels, image_size)
        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        with_label = False if self.cv == 0 else True
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=False, with_label=with_label)
        image_size = ImageSizeGenerator(self.dim, [None] * 3, self.image_spacing, valid_output_sizes=[self.valid_output_sizes_x, self.valid_output_sizes_y, self.valid_output_sizes_z], name='output_size', kwparents={'extent': sources['landmarks_bb_extent']})
        reference_transformation = self.spatial_transformation(sources, image_size)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing, self.smooth_labels, image_size, with_label=with_label)
        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)

