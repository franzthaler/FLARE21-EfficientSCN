# FLARE21-EfficientSCN
Our implementation used for the [MICCAI 2021 FLARE Challenge](https://flare.grand-challenge.org/) titled **Efficient Multi-Organ Segmentation Using SpatialConfiguartion-Net with Low GPU Memory Requirements**.

You need to have the [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) framework by [Christian Payer](https://github.com/christianpayer) downloaded and in your PYTHONPATH for the scripts to work.

If you have questions about the code, write me a [mail](mailto:franz.thaler@icg.tugraz.at).


## Dependencies
The following frameworks/libraries were used in the version as stated. If you run into problems with the libraries, please verify that you have the same version installed.

- Python 3.9
- TensorFlow 2.6
- SimpleITK 2.0
- Numpy 1.20


## Dataset and Preprocessing
The dataset as well as a detailed description of it can be found on the [challenge website](https://flare.grand-challenge.org/Data/). Follow the steps described there to download the data.

Define the `base_dataset_folder` containing the downloaded `TrainingImg`, `TrainingMask` and `ValidationImg` in the script `preprocessing/preprocessing.py` and execute it to generate `TrainingImg_small` and `TrainingMask_small`.

Also, download the `setup` folder provided in this repository and place it in the `base_dataset_folder`, the following structure is expected:

    .                                       # The `base_dataset_folder` of the dataset
    ├── TrainingImg                         # Image folder containing all training images
    │   ├── train_000_0000.nii.gz            
    │   ├── ...                   
    │   └── train_360_0000.nii.gz            
    ├── TrainingMask                        # Image folder containing all training masks
    │   ├── train_000.nii.gz            
    │   ├── ...                   
    │   └── train_360.nii.gz  
    ├── ValidationImg                       # Image folder containing all validation images
    │   ├── validation_000_0000.nii.gz            
    │   ├── ...                   
    │   └── validation_360_0000.nii.gz  
    ├── TrainingImg_small                   # Image folder containing all downsampled training images generated by `preprocessing/preprocessing.py`
    │   ├── train_000_0000.nii.gz            
    │   ├── ...                   
    │   └── train_360_0000.nii.gz  
    ├── TrainingMask_small                  # Image folder containing all downsampled training masks generated by `preprocessing/preprocessing.py`
    │   ├── train_000_0000.nii.gz            
    │   ├── ...                   
    │   └── train_360_0000.nii.gz  
    └── setup                               # Setup folder as provided in this repository


## Train Models
To train a localization model, run `localization/main.py` after defining the `base_dataset_folder` as well as the `base_output_folder`.

To train a segmentation model, run `scn/main.py`. Again, `base_dataset_folder` and `base_output_folder` need to be set accordingly beforehand.

In both cases in function `run()`, the variable `cv` can be set to 0, 1, 2, 3 or 4. The values 1-4 represent the respective cross-validation fold. When choosing 0, all training data is used to train the model, which also deactivates the generation of test outputs.

Further parameters like the number of training iterations (`max_iter`) and the number of iterations after which to perfrom testing (`test_iter`) can be modified in `__init__()` of the `MainLoop` class.


## Generate a SavedModel
To convert a trained network to a SavedModel, the script `localization/main_create_model.py` respectively `scn/main_create_model.py` can be used after a model was trained.

Before running the respective script, the variable `load_model_base` needs to be set to the trained models output folder, e.g., `.../localization/cv1/2021-09-27_13-18-59`.

Furthermore, `load_model_iter` should be set to the same value as `max_iter` used during training the model. The value needs to be set to an iteration for which the network weights have been generated.


## Generate tf_utils_module
The script `inference/inference_tf_utils_module.py` can be used to trace and save the tf.functions used for preprocessing during inference into a SavedModel and generate `saved_models/tf_utils_module`.

To do so, the `input_path` and `output_path` need to be defined in the script.
The `input_path` is expected to contain valid images, we suggest to use the folder `ValidationImg`.


## Inference
The provided inference script can be used to evaluate the performance of our method on unseen data efficiently.

The script `inference/inference.py` requires that all SavedModels are present in the `saved_models` folder, i.e., `saved_models/localization`, `saved_models/segmentation` and `saved_models/tf_utils_module` need to contain the respective SavedModel. Either, use the provided SavedModels for inference by copying them from `submitted_saved_models` to `saved_models`, or use your own models generated as described above.

Additionally, the `input_path` and `output_path` need to be defined in the script.
The `input_path` is expected to contain valid images, we suggest to use the folder `ValidationImg`.


    .                                       # The base folder of this repository.
    ├── saved_models                        # Required by `inference.py`.
    │   ├── localization                    # SavedModel of the localization model.
    │   │   ├── assets
    │   │   ├── variables
    │   │   └── saved_model.pb
    │   ├── segmentation                    # SavedModel of the segmentation (scn) model.
    │   │   ├── assets
    │   │   ├── variables
    │   │   └── saved_model.pb
    │   └── tf_utils_module                 # SavedModel of the tf.functions used for preprocessing during inference.
    │       ├── assets
    │       ├── variables
    │       └── saved_model.pb
    ...


## Docker
The provided `Dockerfile` can be used to generate a docker image which can readily be used for inference.
The SavedModels are expected in the folder `saved_models`, either copy the provided SavedModels from `submitted_saved_models` to `saved_models` or generate your own.
If you have a problem with setting up docker, please refer to the [documentation](https://docs.docker.com/).

To build a docker model, run the following command in the folder containing the `Dockerfile`.

```
docker build -t icg .
```

To run your built docker, use the command below, after defining the input and output directories within the command.
We recommend to use `ValidationImg` as input folder.

If you have multiple GPUs and want to select a specific one to run the docker image, modify `/dev/nvidia0` to the respective GPUs identifier, e.g., `/dev/nvidia1`.

```
docker container run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidiactl --name icg --rm -v /PATH/TO/DATASET/ValidationImg/:/workspace/inputs/ -v /PATH/TO/OUTPUT/FOLDER/:/workspace/outputs/ icg:latest /bin/bash -c "sh predict.sh" 
```





