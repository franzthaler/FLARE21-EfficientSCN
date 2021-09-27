#!/usr/bin/python
import os
import tensorflow as tf

from tensorflow_train_v2.losses.semantic_segmentation_losses import generalized_dice_loss
from main import MainLoop
import utils
from network import Unet, UnetAvgLinear3D


class CustomModule(tf.Module):
    def __init__(self, model, data_format='channels_first'):
        super(CustomModule, self).__init__()
        self.model = model
        self.data_format = data_format
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1

    @tf.function(input_signature=[tf.TensorSpec([None, 1, None, None, None], tf.float16)])  # mixed precision
    def __call__(self, image, training=False):
        prediction = self.model(image, training=training)
        label = prediction[:, :1, ...] < prediction[:, 1:, ...]
        return label


def run():
    network_parameters = {'num_filters_base': 32,
                          'num_levels': 5,
                          'dropout_ratio': 0.1,
                          'activation': 'lrelu',
                          }

    cached_datasource = False
    dataset_threads = 8

    # TODO set model path and iter to load
    load_model_base = '/SET/PATH/TO/A/FINISHED/EXPERIMENT'
    load_model_iter = 100000  # set this to the final or an intermediate number of iterations for which model weights where saved

    load_model_filename = os.path.join(load_model_base, 'weights', f'ckpt-{load_model_iter}')

    output_folder_name = f'localization_create_module/'
    loop = MainLoop(cv=0,
                    network=Unet,
                    unet=UnetAvgLinear3D,
                    normalized_prediction=False,
                    loss=generalized_dice_loss,
                    network_parameters=network_parameters,
                    learning_rate=1e-5,
                    cached_datasource=cached_datasource,
                    dataset_threads=dataset_threads,
                    output_folder_name=output_folder_name,
                    load_model_filename=load_model_filename)

    loop.init_model()
    loop.init_optimizer()
    loop.init_output_folder_handler()
    loop.init_checkpoint()
    loop.init_checkpoint_manager()
    if loop.load_model_filename is not None:
        loop.load_model()

    module_folder = '../saved_models/localization'
    module = CustomModule(loop.model)

    data_format = loop.data_format
    if data_format == 'channels_first':
        image = tf.zeros([1, 1, 64, 64, 64], dtype=tf.float16)
    else:
        image = tf.zeros([1, 64, 64, 64, 1], dtype=tf.float16)
    prediction = module(image)  # forward pass to trace model such that it can be saved

    tf.saved_model.save(module, module_folder)


if __name__ == '__main__':
    run()

