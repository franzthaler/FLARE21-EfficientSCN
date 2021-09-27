import os
import glob

import concurrent.futures
import numpy as np
import tensorflow as tf
import SimpleITK as sitk


def compute_segmentation_spacing_from_bbox(bbox_start, bbox_end, valid_sizes_for_segmentation, segmentation_spacing_base_value, segmentation_spacing_tolerance):
    bbox_extent = bbox_end - bbox_start
    max_valid_sizes_for_segmentation = tf.Variable([max(x) for x in valid_sizes_for_segmentation], dtype=tf.float32)
    computed_spacing = bbox_extent / max_valid_sizes_for_segmentation
    max_spacing_from_bb = float(tf.reduce_max(computed_spacing).numpy())
    if max_spacing_from_bb > segmentation_spacing_base_value - segmentation_spacing_tolerance:  # get 5% extra space on the borders
        segmentation_spacing = [max_spacing_from_bb + segmentation_spacing_tolerance] * 3
        segmentation_spacing_tf = tf.convert_to_tensor(list(reversed(segmentation_spacing)))
    else:
        segmentation_spacing = [segmentation_spacing_base_value] * 3
        segmentation_spacing_tf = tf.convert_to_tensor(list(reversed(segmentation_spacing)))
    return segmentation_spacing, segmentation_spacing_tf


def compute_expected_localization_size_and_correct_spacing(size, spacing, localization_spacing, valid_sizes_for_localization):
    expected_size = np.ceil(np.array(size) * spacing / localization_spacing)
    localization_max_size = [max(x) for x in valid_sizes_for_localization]
    max_ratio = max(expected_size / localization_max_size)
    max_ratio += 0.02  # additional percentage to make sure that the image is going to fit
    if max_ratio > 1:
        localization_spacing = [x * max_ratio for x in localization_spacing]
    localization_spacing_tf = tf.convert_to_tensor(list(reversed(localization_spacing)), tf.float32)
    return localization_spacing, localization_spacing_tf


def infer(images_list,
          input_path,
          output_path,
          file_extension,
          utils_module,
          localization_module,
          segmentation_module,
          future_utils_module,
          future_localization_module,
          future_segmentation_module,
          data_format):

    assert utils_module or future_utils_module, 'utils_module and future_utils_module are None'
    assert localization_module or future_localization_module, 'localization_module and future_localization_module are None'
    assert segmentation_module or future_segmentation_module, 'segmentation_module and future_segmentation_module are None'

    # parameters localization
    valid_sizes_for_localization = [[32, 48, 64, 80], [32, 48, 64, 80], list(range(32, 257, 16))]
    valid_sizes_for_localization_tf = list(reversed(valid_sizes_for_localization))
    valid_sizes_for_localization_x = tf.convert_to_tensor(valid_sizes_for_localization_tf[0])
    valid_sizes_for_localization_y = tf.convert_to_tensor(valid_sizes_for_localization_tf[1])
    valid_sizes_for_localization_z = tf.convert_to_tensor(valid_sizes_for_localization_tf[2])
    localization_spacing = [6.0] * 3
    localization_sigma_tf = tf.convert_to_tensor(0.0)

    # parameters segmentation
    valid_sizes_for_segmentation = [[32, 64, 96, 128, 160], [32, 64, 96, 128], [32, 64, 96, 128, 160]]
    valid_sizes_for_segmentation_x = tf.convert_to_tensor(valid_sizes_for_segmentation[0])
    valid_sizes_for_segmentation_y = tf.convert_to_tensor(valid_sizes_for_segmentation[1])
    valid_sizes_for_segmentation_z = tf.convert_to_tensor(valid_sizes_for_segmentation[2])
    segmentation_spacing_base_value = 2.0
    segmentation_spacing_tolerance = 0.1
    segmentation_sigma_tf = tf.convert_to_tensor(1.0)

    next_future_image = None
    for filename_i, input_filename in enumerate(images_list):
        # get the next input image which was loaded in parallel
        if next_future_image:
            image = next_future_image.result()
        else:
            image = sitk.ReadImage(os.path.join(input_path, input_filename + file_extension), sitk.sitkInt16)

        # start loading the next input image if one exists
        if filename_i+1 != len(images_list):
            next_input_filename = images_list[filename_i+1]
            next_future_image = Future(load_image,
                                       dict(image_path=os.path.join(input_path, next_input_filename + file_extension),
                                            dtype=sitk.sitkInt16))

        output_filename = input_filename.replace('_0000', '')
        old_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
        new_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines([1, 0, 0, 0, 1, 0, 0, 0, 1])
        image = sitk.DICOMOrient(image, new_orientation)
        size = image.GetSize()
        spacing = image.GetSpacing()
        # use average spacing if spacing from image is 1x1x1
        if all([x == 1 for x in spacing]):
            spacing = [0.814528640267884, 0.814528640267884, 2.52795318268203]
        spacing_tf = tf.convert_to_tensor(list(reversed(spacing)))

        image_np = sitk.GetArrayViewFromImage(image)
        if data_format == 'channels_first':
            image_tf = tf.convert_to_tensor(image_np, dtype=tf.float16)[None, None, ...]
        else:
            image_tf = tf.convert_to_tensor(image_np, dtype=tf.float16)[None, ..., None]

        # use different spacing for localization preprocessing if image does not fit
        localization_spacing, localization_spacing_tf = compute_expected_localization_size_and_correct_spacing(size, spacing, localization_spacing, valid_sizes_for_localization)

        if not utils_module:
            utils_module = future_utils_module.result()
        localization_input, localization_paddings = utils_module.preprocess_for_localization(image_tf, spacing_tf,
                                                                                             localization_spacing_tf,
                                                                                             localization_sigma_tf,
                                                                                             valid_sizes_for_localization_x,
                                                                                             valid_sizes_for_localization_y,
                                                                                             valid_sizes_for_localization_z
                                                                                             )

        if not localization_module:
            localization_module = future_localization_module.result()

        localization_output = localization_module(localization_input)

        bbox_end, bbox_start = utils_module.bbox_from_localization_output(localization_output, localization_spacing_tf, localization_paddings)

        # increase segmentation spacing if bbox does not fit
        segmentation_spacing, segmentation_spacing_tf = compute_segmentation_spacing_from_bbox(bbox_start, bbox_end, valid_sizes_for_segmentation, segmentation_spacing_base_value, segmentation_spacing_tolerance)

        segmentation_input, segmentation_input_size, croppings, paddings = utils_module.preprocess_for_segmentation(
            image_tf,
            spacing_tf,
            segmentation_spacing_tf,
            bbox_end,
            bbox_start,
            segmentation_sigma_tf,
            valid_sizes_for_segmentation_x,
            valid_sizes_for_segmentation_y,
            valid_sizes_for_segmentation_z,
        )

        if not segmentation_module:
            segmentation_module = future_segmentation_module.result()

        segmentation_output_wo_argmax = segmentation_module(segmentation_input)

        segmentation_output_resampled = utils_module.postprocess_for_segmentation(segmentation_output_wo_argmax,
                                                                                  paddings, croppings,
                                                                                  segmentation_input_size,
                                                                                  )

        segmentation_output_sitk = sitk.GetImageFromArray(tf.squeeze(segmentation_output_resampled.numpy()))
        segmentation_output_sitk.CopyInformation(image)
        segmentation_output_sitk = sitk.DICOMOrient(segmentation_output_sitk, old_orientation)
        sitk.WriteImage(segmentation_output_sitk, os.path.join(output_path, output_filename + file_extension))


def load_saved_module(saved_module_folder):
    return tf.saved_model.load(saved_module_folder)


def load_image(image_path, dtype=sitk.sitkInt16):
    return sitk.ReadImage(image_path, dtype)


class Future:
    def __init__(self, function, args):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future = self.executor.submit(function, **args)

    def result(self):
        return self.future.result()


def set_memory_growth():
    # set memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)


def main():
    set_memory_growth()

    # TODO set input and output path
    input_path = '/PATH/TO/INPUT/IMAGES/TO/USE'
    output_path = '/OUTPUT/PATH'

    localization_module_folder = 'saved_models/localization'
    segmentation_module_folder = 'saved_models/segmentation'
    utils_module_folder = 'saved_models/tf_utils_module'

    # NOTE: add additional output folder
    from datetime import datetime as dt
    output_path = os.path.join(output_path, dt.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_format = 'channels_first'
    file_extension = '.nii.gz'

    images_list = sorted([os.path.basename(x).replace(file_extension, '') for x in glob.glob(os.path.join(input_path, f'*{file_extension}'))])

    future_utils_module = Future(load_saved_module, dict(saved_module_folder=utils_module_folder))
    future_localization_module = Future(load_saved_module, dict(saved_module_folder=localization_module_folder))
    future_segmentation_module = Future(load_saved_module, dict(saved_module_folder=segmentation_module_folder))

    infer(images_list=images_list,
          input_path=input_path,
          output_path=output_path,
          file_extension=file_extension,
          utils_module=None,
          localization_module=None,
          segmentation_module=None,
          future_utils_module=future_utils_module,
          future_localization_module=future_localization_module,
          future_segmentation_module=future_segmentation_module,
          data_format=data_format)


if __name__ == '__main__':
    main()

