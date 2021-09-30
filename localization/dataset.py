import os

import numpy as np
import SimpleITK as sitk
from datasets.graph_dataset import GraphDataset

from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.image_size_generator import ImageSizeGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.gamma import change_gamma_unnormalized
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from utils.random import float_uniform
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
                 valid_output_sizes,
                 cache_maxsize=8192,
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=1.0,
                 use_landmarks=True,
                 num_labels=5,
                 merge_foreground_labels=False,
                 displacement_field_sampling_factor=2,
                 image_folder=None,
                 setup_folder=None,
                 image_filename_postfix='_0000',
                 image_filename_extension='.nii.gz',
                 labels_filename_postfix='',
                 labels_filename_extension='.nii.gz',
                 landmark_file_postfix='',
                 landmarks_file=None,
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
        self.valid_output_sizes = valid_output_sizes
        self.base_folder = base_folder
        self.cv = cv
        self.cached_datasource = cached_datasource
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.use_landmarks = use_landmarks
        self.num_labels = num_labels
        self.merge_foreground_labels = merge_foreground_labels
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

        self.image_base_folder = image_folder or os.path.join(self.base_folder, 'TrainingImg_small')
        self.label_base_folder = image_folder or os.path.join(self.base_folder, 'TrainingMask_small')
        self.setup_base_folder = setup_folder or os.path.join(self.base_folder, 'setup')
        self.landmarks_file = landmarks_file or os.path.join(self.setup_base_folder, f'landmark{landmark_file_postfix}.csv')
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

    def datasources(self, iterator, preprocessing, cached, with_label=True, landmarks_exist=True):
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

        if landmarks_exist:
            landmark_datasource = LandmarkDataSource(self.landmarks_file, 1, self.dim, name='landmarks', parents=[iterator])
            datasource_dict['landmarks'] = landmark_datasource

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

    def data_generators(self, datasources, transformation, image_post_processing, mask_post_processing, with_label=True, key_suffix='', image_size=None):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: The datasources dictionary (see self.datasources()).
        :param transformation: The spatial transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        generator_dict = {}
        image_generator = ImageGenerator(self.dim, None, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format, name=f'image{key_suffix}', kwparents={'image': datasources['image'], 'transformation': transformation, 'output_size': image_size}, resample_default_pixel_value=self.background_pixel_value)
        generator_dict[f'data{key_suffix}'] = image_generator
        if with_label:
            mask_image_generator = ImageGenerator(self.dim, None, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, data_format=self.data_format, name=f'labels{key_suffix}', kwparents={'image': datasources['labels'], 'transformation': transformation, 'output_size': image_size})
            generator_dict[f'labels{key_suffix}'] = mask_image_generator
        return generator_dict

    def datasource_preprocessing_train(self, image):
        if self.input_gaussian_sigma > 0.0:
            image = gaussian_sitk(image, self.input_gaussian_sigma)
        # shrink image size such that datasource spacing is < than image_spacing
        shrink_factor = [max(int(x // y), 1) for x, y in zip(self.image_spacing, list(image.GetSpacing()))]
        image = sitk.Shrink(image, shrink_factor)
        return image

    def datasource_preprocessing_val(self, image):
        if self.input_gaussian_sigma > 0.0:
            image = gaussian_sitk(image, self.input_gaussian_sigma)
        return image

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        if self.merge_foreground_labels:
            split = split_label_image(np.squeeze(image, 0), list(range(5)), np.uint8)
            split = [split[0], np.clip(np.sum(split[1:], axis=0, dtype=np.uint8), 0, 1)]
        else:
            split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
        if self.label_gaussian_sigma > 0.0:
            # if fg-bg segmentation, compute gaussian smoothing only for fg, compute bg as 1 - fg
            if len(split) == 2:
                fg_smoothed = gaussian(split[1], self.label_gaussian_sigma)
                bg_smoothed = np.ones_like(fg_smoothed) - fg_smoothed
                split_smoothed = [bg_smoothed, fg_smoothed]
            else:
                split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
        else:
            split_smoothed = split
        smoothed = np.stack(split_smoothed, 0)
        image_smoothed = np.argmax(smoothed, axis=0)
        split = split_label_image(image_smoothed, list(range(self.num_labels)), np.uint8)
        return np.stack(split, 0)

    def set_spacing_if_1x1x1(self, image):
        spacing = list(image.GetSpacing())
        if all(x == 1 for x in spacing):
            # mean of FLARE21 dataset
            mean = [0.814528640267884, 0.814528640267884, 2.52795318268203]
            image.SetSpacing(mean)
        return image

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        random_lambda = float_uniform(0.8, 1.2)
        image = change_gamma_unnormalized(image, random_lambda)
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

    def spatial_transformation_augmented(self, datasources, image_size=None):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.use_landmarks:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing),
                                    deformation.Output(self.dim, [8, 8, 8], 15, image_size, self.image_spacing, spline_order=2),
                                    ])
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return DisplacementField(output_size=None, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, name='image', parents=[comp], kwparents=kwparents)


    def spatial_transformation(self, datasources, image_size=None, landmarks_exist=True):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.use_landmarks and landmarks_exist:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        with_label = True
        iterator = IdListIterator(self.train_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, preprocessing=self.datasource_preprocessing_train, cached=self.cached_datasource, with_label=with_label)
        if self.valid_output_sizes is not None:
            image_size = ImageSizeGenerator(self.dim, [None] * self.dim, self.image_spacing, valid_output_sizes=self.valid_output_sizes, kwparents={'image': sources['image']})
        else:
            image_size = self.image_size
        reference_transformation = self.spatial_transformation_augmented(sources, image_size=image_size)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.split_labels, with_label=with_label, image_size=image_size)
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
        landmarks_exist = False if self.cv == 0 else True
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, preprocessing=self.datasource_preprocessing_val, cached=False, with_label=with_label, landmarks_exist=landmarks_exist)
        if self.valid_output_sizes is not None:
            image_size = ImageSizeGenerator(self.dim, [None] * self.dim, self.image_spacing, valid_output_sizes=self.valid_output_sizes, kwparents={'image': sources['image']})
        else:
            image_size = self.image_size
        reference_transformation = self.spatial_transformation(sources, image_size=image_size, landmarks_exist=landmarks_exist)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing, self.split_labels, with_label=with_label, image_size=image_size)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
