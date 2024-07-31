import os
import sys

import torch
from matplotlib import pyplot as plt
from torch import optim

from utils.log_util import log_and_print, log


def get_os_dependent_paths(model_ver, partition):
    assert (partition == 'train' or partition == 'test')
    if sys.platform == 'darwin':  # mac
        data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
        list_path = os.path.join(data_dir, '{}/list_{}.txt'.format(partition, sys.platform))
        save_path = './model_{}'.format(model_ver)
    elif sys.platform == 'win32':  # windows
        data_dir = 'C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\synth_datasets'
        list_path = os.path.join(data_dir, '{}\\list_{}.txt'.format(partition, sys.platform))
        save_path = '.\\model_{}'.format(model_ver)
    else:  # ubuntu
        data_dir = '/mnt/storage_1/bell_5g_datasets/synth_datasets'
        list_path = os.path.join(data_dir, '{}/list_{}.txt'.format(partition, sys.platform))
        save_path = './model_{}'.format(model_ver)
    return list_path, save_path


def print_hyperparams(**kwargs):
    log_and_print('Hyperparameters:')
    for key, value in kwargs.items():
        log_and_print('\t{}: {}'.format(key, str(value)))


def print_metric_plots(metrics_history, model_ver, save_path):
    for name, m_train, m_val in metrics_history:
        plt.clf()
        plt.plot(m_train)
        plt.plot(m_val)
        plt.title("Training {}".format(name))
        plt.ylabel(name)
        plt.xlabel("epoch")
        plt.legend(['train', 'val'])
        plt.savefig(os.path.join(save_path, 'model_{}_train_{}_plot.png'.format(model_ver, name)))
