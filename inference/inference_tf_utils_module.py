import itertools

from tensorflow_train_v2.utils.data_format import get_image_size
from tensorflow_train_v2.utils.resize import resize
from inference import *
from tensorflow_train_v2.utils.data_format import get_tf_data_format, get_channel_index, get_channel_size


class CustomModule(tf.Module):
    def __init__(self, data_format='channels_first'):
        super(CustomModule, self).__init__()
        self.data_format = data_format
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float16),
                                  tf.TensorSpec(None, tf.int32)])
    def gaussian_kernel1d(self, sigma, filter_shape):
        """
        Calculate a 1d gaussian kernel.
        """
        sigma = tf.convert_to_tensor(sigma)
        coordinates = tf.cast(tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1), sigma.dtype)
        kernel = tf.exp(-0.5 / (sigma ** 2) * coordinates ** 2)
        kernel = kernel / tf.math.reduce_sum(kernel)
        return kernel

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16),
                                  tf.TensorSpec(None, tf.float32),
                                  ])
    def gaussian(self,
                 image,
                 sigma):
        """
        Gaussian filtering of a 2d or 3d image tensor.
        :param image: The tf image tensor to filter.
        :param sigma: The sigma per dimension. If only a single sigma is given, use same sigma for all dimensions.
        :param filter_shape: The shape of the filter. If None, use sigma to calculate filter shape.
        :param padding: The padding to use before filtering.
        :param constant_values: If padding is constant, use this value for padding.
        :param data_format: 'channels_first' or 'channels_last'
        :param name: The name of the tf operation. If None, use 'gaussian'.
        """
        filter_shape = None
        padding = 'symmetric'
        constant_values = 0
        name = None
        with tf.name_scope(name or 'gaussian'):
            image = tf.convert_to_tensor(image, name='image')
            dim = image.shape.ndims - 2
            sigma = tf.convert_to_tensor(sigma, name='sigma')
            if sigma.shape.ndims == 0:
                sigma = tf.stack([sigma] * dim)

            if filter_shape is not None:
                filter_shape = tf.convert_to_tensor(filter_shape, name='filter_shape', dtype=tf.int32)
            else:
                filter_shape = tf.cast(tf.math.ceil(tf.cast(sigma, tf.float32) * 4 + 0.5) * 2 + 1, tf.int32)

            # calculate later needed tensor values (must be done before padding!)
            data_format_tf = get_tf_data_format(image, data_format=self.data_format)
            channel_size = get_channel_size(image, data_format=self.data_format, as_tensor=False)
            channel_axis = get_channel_index(image, data_format=self.data_format)

            # Keep the precision if it's float;
            # otherwise, convert to float32 for computing.
            orig_dtype = image.dtype
            if not image.dtype.is_floating:
                image = tf.cast(image, tf.float32)

            # calculate gaussian kernels
            sigma = tf.cast(sigma, image.dtype)
            gaussian_kernels = []
            for i in range(dim):
                current_gaussian_kernel = self.gaussian_kernel1d(sigma[i], filter_shape[i])
                current_gaussian_kernel = tf.reshape(current_gaussian_kernel, [-1 if j == i else 1 for j in range(dim + 2)])
                gaussian_kernels.append(current_gaussian_kernel)

            # pad image for kernel size
            paddings_half = filter_shape // 2
            if self.data_format == 'channels_first':
                paddings_half = tf.concat([tf.zeros([2], tf.int32), paddings_half], axis=0)
            else:
                paddings_half = tf.concat([tf.zeros([1], tf.int32), paddings_half, tf.zeros([1], tf.int32)], axis=0)
            paddings = tf.stack([paddings_half, paddings_half], axis=1)
            image = tf.pad(image, paddings, mode=padding, constant_values=constant_values)

            # channelwise convolution
            split_inputs = tf.split(image, channel_size, axis=channel_axis, name='split')
            output_list = []
            for i in range(len(split_inputs)):
                current_output = split_inputs[i]
                for current_gaussian_kernel in gaussian_kernels:
                    if dim == 2:
                        current_output = tf.nn.conv2d(current_output, current_gaussian_kernel, (1, 1, 1, 1), data_format=data_format_tf, name='conv' + str(i), padding='VALID')
                    else:
                        current_output = tf.nn.conv3d(current_output, current_gaussian_kernel, (1, 1, 1, 1, 1), data_format=data_format_tf, name='conv' + str(i), padding='VALID')
                output_list.append(current_output)
            output = tf.concat(output_list, axis=channel_axis, name='concat')

            return tf.cast(output, orig_dtype)

    @tf.function
    def pad_image(self, image, paddings, pad_value=0, data_format='channels_last'):
        if data_format == 'channels_last':
            paddings_all_dims = tf.concat([[[0, 0]], paddings, [[0, 0]]], axis=0)
        else:
            paddings_all_dims = tf.concat([[[0, 0], [0, 0]], paddings], axis=0)
        return tf.pad(image, paddings_all_dims, constant_values=pad_value)

    @tf.function
    def crop_image(self, image, croppings, data_format='channels_last'):
        image_size = get_image_size(image, data_format=data_format, as_tensor=True)
        cropping_slices = [slice(croppings[i, 0], image_size[i] - croppings[i, 1]) for i in range(3)]
        if data_format == 'channels_last':
            cropping_slices = [slice(None)] + cropping_slices + [slice(None)]
        else:
            cropping_slices = [slice(None), slice(None)] + cropping_slices
        return image[tuple(cropping_slices)]

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec(None, tf.float32),
                                  ])
    def smooth_image(self, image, spacing, sigma):
        sigma_with_spacing = sigma / spacing
        return self.gaussian(image, sigma_with_spacing)

    @tf.function
    def resize_image(self, image, new_size, interpolator='linear', data_format='channels_last'):
        return resize(image, output_size=new_size, interpolator=interpolator, data_format=data_format)

    @tf.function
    def size_for_spacing(self, size, spacing, new_spacing):
        new_image_size = tf.cast(size, tf.float32) * spacing / new_spacing
        return tf.cast(tf.math.ceil(new_image_size), tf.int32)


    @tf.function(input_signature=[tf.TensorSpec([3], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  ])
    def get_valid_size(self, size,
                       valid_output_sizes_x, valid_output_sizes_y, valid_output_size_z
                       ):
        output_size = []
        valid_size = tf.reduce_min(tf.where(valid_output_sizes_x > size[0], valid_output_sizes_x, valid_output_sizes_x[-1]))
        output_size.append(valid_size)

        valid_size = tf.reduce_min(tf.where(valid_output_sizes_y > size[1], valid_output_sizes_y, valid_output_sizes_y[-1]))
        output_size.append(valid_size)

        valid_size = tf.reduce_min(tf.where(valid_output_size_z > size[2], valid_output_size_z, valid_output_size_z[-1]))
        output_size.append(valid_size)
        return tf.stack(output_size)

    @tf.function
    def center_pad_to_size(self, image, size, pad_value=-1024, data_format='channels_last'):
        image_size = get_image_size(image, data_format=data_format, as_tensor=True)
        pad_size = size - image_size
        pad_size_before = pad_size // 2
        pad_size_after = pad_size - pad_size_before
        paddings = tf.stack([[pad_size_before[i], pad_size_after[i]] for i in range(size.shape[0])])
        return self.pad_image(image, paddings, pad_value, data_format=data_format), paddings

    @tf.function
    def intensity_postprocessing_ct(self, image, shift=0, scale=1/2048, clamp_min=-1.0, clamp_max=1.0):
        output_image = image
        output_image += shift
        output_image *= scale
        output_image = tf.clip_by_value(output_image, clamp_min, clamp_max)
        return output_image

    @tf.function
    def bounding_box(self, image):
        """
        Calculate the bounding box of an image of pixels != 0. Both start and end index are inclusive, i.e., contain the image value.
        If the image is all zeroes, return np arrays of np.nan.
        :param image: The image.
        :return: The bounding box as start and end tuple.
        """
        dim = image.shape.ndims
        start = []
        end = []
        for i, ax in enumerate(itertools.combinations(reversed(range(dim)), dim - 1)):
            nonzero = tf.reduce_any(image, axis=ax)
            nonzero_where = tf.cast(tf.where(nonzero), tf.int32)
            if len(nonzero_where) > 0:
                curr_start, curr_end = nonzero_where[0, 0], nonzero_where[-1, 0]
            else:
                curr_start, curr_end = tf.constant(0, tf.int32), tf.cast(tf.shape(image)[i] - 1, tf.int32)
            start.append(curr_start)
            end.append(curr_end)
        return tf.stack(start), tf.stack(end)

    @tf.function
    def crop_and_pad_with_bbox(self, image, spacing, bbox_start, bbox_end, data_format):
        image_size = get_image_size(image, data_format=data_format, as_tensor=True)
        bbox_start_index = tf.cast(tf.math.floor(bbox_start / spacing), tf.int32)
        bbox_start_index_in_image = tf.maximum(bbox_start_index, 0)
        pad_before = bbox_start_index_in_image - bbox_start_index
        bbox_end_index = tf.cast(tf.math.ceil((bbox_end + 1) / spacing), tf.int32)
        bbox_end_index_in_image = tf.minimum(bbox_end_index, image_size)
        pad_after = bbox_end_index - bbox_end_index_in_image
        paddings = tf.stack([pad_before, pad_after], axis=1)
        croppings = tf.stack([bbox_start_index_in_image, image_size - bbox_end_index_in_image], axis=1)

        cropped_image = self.crop_image(image, croppings, data_format=data_format)
        padded_image = self.pad_image(cropped_image, paddings, data_format=data_format)

        return padded_image, paddings, croppings

    @tf.function
    def valid_bbox_extent_for_segmentation(self, bbox_end, bbox_start, spacing,
                                           # valid_sizes,
                                           valid_sizes_x, valid_sizes_y, valid_sizes_z):
        bbox_center = (bbox_start + bbox_end) * 0.5
        bbox_extent_in_segmentation_spacing = tf.cast(tf.math.ceil(bbox_end - bbox_start) / spacing,
                                                      tf.int32)
        valid_bbox_extent_in_segmentation_spacing = self.get_valid_size(bbox_extent_in_segmentation_spacing,
                                                                        # valid_sizes
                                                                        valid_sizes_x, valid_sizes_y, valid_sizes_z)
        valid_bbox_extent = tf.cast(valid_bbox_extent_in_segmentation_spacing, tf.float32) * spacing
        bbox_start = bbox_center - valid_bbox_extent // 2
        bbox_end = bbox_center + valid_bbox_extent // 2
        return bbox_start, bbox_end, valid_bbox_extent_in_segmentation_spacing

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec(None, tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  ])
    def preprocess_for_localization(self, image, spacing, new_spacing, sigma, valid_sizes_x, valid_size_y, valid_size_z):
        image_size = get_image_size(image, data_format=self.data_format, as_tensor=True)
        if sigma > 0.0:
            image_smoothed = self.smooth_image(image, spacing, sigma)
        else:
            image_smoothed = image
        new_size = self.size_for_spacing(image_size, spacing, new_spacing)
        image_resized = self.resize_image(image_smoothed, new_size, interpolator='area', data_format=self.data_format)
        valid_size = self.get_valid_size(new_size, valid_sizes_x, valid_size_y, valid_size_z)
        image_resized_padded, paddings = self.center_pad_to_size(image_resized, valid_size, data_format=self.data_format)
        image_resized_padded = self.intensity_postprocessing_ct(image_resized_padded)
        return image_resized_padded, paddings

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.bool),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec([3, 2], tf.int32),
                                  ])
    def bbox_from_localization_output(self, image, spacing, paddings):
        additional_bbox_padding = 10
        if self.data_format == 'channels_last':
            image = image[0, ..., 0]
        else:
            image = image[0, 0, ...]
        bbox_start, bbox_end = self.bounding_box(image)
        bbox_start -= paddings[:, 0]
        bbox_end -= paddings[:, 0]
        bbox_start = spacing * tf.cast(bbox_start, tf.float32) - additional_bbox_padding
        bbox_end = spacing * tf.cast(bbox_end + 1, tf.float32) + additional_bbox_padding
        return bbox_end, bbox_start

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec([3], tf.float32),
                                  tf.TensorSpec(None, tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  ])
    def preprocess_for_segmentation(self, image, spacing, new_spacing, bbox_end, bbox_start, sigma,
                                    valid_sizes_x, valid_sizes_y, valid_sizes_z
                                    ):
        if sigma > 0.0:
            image_smoothed = self.smooth_image(image, spacing, sigma)
        else:
            image_smoothed = image

        bbox_start, bbox_end, valid_bbox_size_for_segmentation = self.valid_bbox_extent_for_segmentation(bbox_end,
                                                                                                         bbox_start,
                                                                                                         new_spacing,
                                                                                                         valid_sizes_x, valid_sizes_y, valid_sizes_z
                                                                                                         )
        image_smoothed_cropped, paddings, croppings = self.crop_and_pad_with_bbox(image_smoothed, spacing, bbox_start, bbox_end,
                                                                             data_format=self.data_format)
        segmentation_cropped_resampled = self.resize_image(image_smoothed_cropped, valid_bbox_size_for_segmentation,
                                                      interpolator='area', data_format=self.data_format)
        segmentation_cropped_size = get_image_size(image_smoothed_cropped, data_format=self.data_format, as_tensor=True)
        segmentation_cropped_resampled = self.intensity_postprocessing_ct(segmentation_cropped_resampled)
        return segmentation_cropped_resampled, segmentation_cropped_size, croppings, paddings

    @tf.function(input_signature=[tf.TensorSpec([None, 5, None, None, None], tf.float16),
                                  tf.TensorSpec([3, 2], tf.int32),
                                  tf.TensorSpec([3, 2], tf.int32),
                                  tf.TensorSpec([3], tf.int32),
                                  ])
    def postprocess_for_segmentation(self, segmentation_output, paddings, croppings, original_cropped_size):
        segmentation_one_hot_resampled = self.resize_image(segmentation_output, original_cropped_size,
                                              interpolator='linear', data_format=self.data_format)
        segmentation_resampled = tf.expand_dims(tf.cast(tf.argmax(segmentation_one_hot_resampled, axis=self.channel_axis, output_type=tf.int32), dtype=tf.uint8), axis=self.channel_axis)
        segmentation_resampled_cropped = self.crop_image(segmentation_resampled, paddings, data_format=self.data_format)
        segmentation_resampled_cropped_padded = self.pad_image(segmentation_resampled_cropped,
                                                          croppings, data_format=self.data_format)
        return segmentation_resampled_cropped_padded

    @tf.function
    def simulate_localization_output(self, image, spacing, new_spacing, valid_sizes, data_format):
        image_size = get_image_size(image, data_format=data_format, as_tensor=True)
        new_size = self.size_for_spacing(image_size, spacing, new_spacing)
        image_resized = self.resize_image(image, new_size, interpolator='nearest', data_format=data_format)
        valid_size = self.get_valid_size(new_size, valid_sizes)
        image_resized_padded, paddings = self.center_pad_to_size(image_resized, valid_size, data_format=data_format)
        return image_resized_padded

    @tf.function
    def simulate_segmentation_output(self, image, spacing, new_spacing, bbox_end, bbox_start,
                                     valid_sizes, data_format='channels_last'):
        bbox_start, bbox_end, valid_bbox_size_for_segmentation = self.valid_bbox_extent_for_segmentation(bbox_end,
                                                                                                    bbox_start,
                                                                                                    new_spacing,
                                                                                                    valid_sizes)
        image_smoothed_cropped, paddings, croppings = self.crop_and_pad_with_bbox(image, spacing, bbox_start, bbox_end,
                                                                             data_format=data_format)
        segmentation_cropped_resampled = self.resize_image(image_smoothed_cropped, valid_bbox_size_for_segmentation,
                                                      interpolator='nearest', data_format=data_format)
        channel_axis = 1 if data_format == 'channels_first' else -1
        segmentation_cropped_resampled = tf.squeeze(segmentation_cropped_resampled, axis=channel_axis)
        segmentation_cropped_resampled = tf.cast(segmentation_cropped_resampled, tf.uint8)
        num_labels = tf.shape(tf.unique(tf.reshape(segmentation_cropped_resampled, [-1]))[0])[0]
        segmentation_cropped_resampled = tf.one_hot(segmentation_cropped_resampled, depth=num_labels, axis=channel_axis)
        return segmentation_cropped_resampled


if __name__ == '__main__':

    # TODO set input and output path (NOTE: only first image will be processed)
    input_path = '/PATH/TO/INPUT/IMAGES/TO/USE'
    output_path = '/OUTPUT/PATH'

    segmentation_module_folder = '../saved_models/segmentation'
    localization_module_folder = '../saved_models/localization'

    # NOTE: add additional output folder
    from datetime import datetime as dt

    output_path = os.path.join(output_path, dt.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_format = 'channels_first'
    file_extension = '.nii.gz'
    future_localization_module = Future(load_saved_module,
                                        dict(saved_module_folder=localization_module_folder))
    future_segmentation_module = Future(load_saved_module,
                                        dict(saved_module_folder=segmentation_module_folder))
    images_list = sorted([os.path.basename(x).replace(file_extension, '') for x in glob.glob(os.path.join(input_path, f'*{file_extension}'))])
    images_list = [images_list[0]]  # NOTE: only one image is required

    # create module
    module_folder = '../saved_models/tf_utils_module'
    utils_module = CustomModule(data_format=data_format)

    # initialize module
    infer(images_list=images_list,
              input_path=input_path,
              output_path=output_path,
              file_extension=file_extension,
              utils_module=utils_module,
              localization_module=None,
              segmentation_module=None,
              future_utils_module=None,
              future_localization_module=future_localization_module,
              future_segmentation_module=future_segmentation_module,
              data_format=data_format)

    # save module
    tf.saved_model.save(utils_module, module_folder)


