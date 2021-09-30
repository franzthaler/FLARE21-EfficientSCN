import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def correct_min_max_order(min_point, max_point):
    for i, (min_v, max_v) in enumerate(zip(min_point, max_point)):
        if min_v > max_v:
            min_point[i] = max_v
            max_point[i] = min_v
    return min_point, max_point


def generate_landmarks_bb_from_label(base_dir):
    output_dir = os.path.join(base_dir, 'setup')
    create_dir(output_dir)
    data_to_write = []
    cur_dir = os.path.join(base_dir, 'TrainingMask')

    cur_file_list = sorted([x for x in os.listdir(cur_dir)])
    for cur_file in tqdm(cur_file_list, 'IDs'):
        cur_id = cur_file.split('.')[0]
        image_sitk = sitk.ReadImage(os.path.join(cur_dir, cur_file))
        image_np = sitk.GetArrayFromImage(image_sitk)
        image_np = image_np.transpose([2, 1, 0])
        image_np = np.where(image_np > 0, 1, 0)
        min_point = [min(np.where(image_np == 1)[i]) for i in range(3)]
        max_point = [max(np.where(image_np == 1)[i]) for i in range(3)]
        min_point_physical = list(image_sitk.TransformIndexToPhysicalPoint([int(x) for x in min_point]))
        max_point_physical = list(image_sitk.TransformIndexToPhysicalPoint([int(x) for x in max_point]))
        min_point_physical, max_point_physical = correct_min_max_order(min_point_physical, max_point_physical)
        data_to_write.append([cur_id] + list(min_point_physical) + list(max_point_physical))

    df = pd.DataFrame(data_to_write)
    df.to_csv(os.path.join(output_dir, f'landmark_bb.csv'), header=False, index=False)


def shrink_images_and_labels(base_dir):
    localization_spacing = [6, 6, 6]
    image_dir = os.path.join(base_dir, 'TrainingImg')
    label_dir = os.path.join(base_dir, 'TrainingMask')
    output_image_dir = os.path.join(base_dir, 'TrainingImg_small')
    output_label_dir = os.path.join(base_dir, 'TrainingMask_small')
    create_dir(output_image_dir)
    create_dir(output_label_dir)
    image_suffix = '_0000'

    image_file_list = sorted([x for x in os.listdir(image_dir)])
    label_file_list = sorted([x for x in os.listdir(label_dir)])
    for cur_file in tqdm(image_file_list, 'IDs (image)'):
        cur_id = cur_file.split('.')[0].replace(image_suffix, '')
        image_sitk = sitk.ReadImage(os.path.join(image_dir, cur_file))
        spacing = list(image_sitk.GetSpacing())
        shrink_factor = []
        for ls, s in zip(localization_spacing, spacing):
            if ls / s > 2.0:
                shrink_factor.append(2)
            else:
                shrink_factor.append(1)
        small_image_sitk = sitk.Shrink(image_sitk, shrink_factor)
        sitk.WriteImage(small_image_sitk, os.path.join(output_image_dir, f'{cur_id}{image_suffix}.nii.gz'))

    for cur_file in tqdm(label_file_list, 'IDs (label)'):
        cur_id = cur_file.split('.')[0]
        image_sitk = sitk.ReadImage(os.path.join(label_dir, cur_file))
        spacing = list(image_sitk.GetSpacing())
        shrink_factor = []
        for ls, s in zip(localization_spacing, spacing):
            if ls / s > 2.0:
                shrink_factor.append(2)
            else:
                shrink_factor.append(1)
        small_image_sitk = sitk.Shrink(image_sitk, shrink_factor)
        sitk.WriteImage(small_image_sitk, os.path.join(output_label_dir, f'{cur_id}.nii.gz'))


def main():
    # TODO set dataset and output folder
    base_dataset_folder = '/SET/PATH/TO/DATASET'

    generate_landmarks_bounding_box = False
    generate_shrinked_images = True

    if generate_landmarks_bounding_box:
        generate_landmarks_bb_from_label(base_dataset_folder)
    if generate_shrinked_images:
        shrink_images_and_labels(base_dataset_folder)


if __name__ == "__main__":
    main()

