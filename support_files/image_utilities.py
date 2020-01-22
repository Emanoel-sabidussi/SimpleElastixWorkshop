from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def get_data_information(weighted_image):

    if isinstance(weighted_image, torch.Tensor):
        img_shape = weighted_image.size()
        img_elem = weighted_image.numel()
        img_dim = len(weighted_image.size())
    else:
        img_shape = np.shape(weighted_image)
        img_elem = np.size(weighted_image)
        img_dim = len(np.shape(weighted_image))

    return [img_shape, img_elem, img_dim]


def get_initial_parameters(observed_series, parameters):
    variables = {}

    if parameters["contrast"] == "T1":
        q_pred_map = Variable(torch.ones_like(observed_series[0]).to(device=parameters["device"]), requires_grad=True)
        a_pred_map = Variable(torch.ones_like(observed_series[0]).to(device=parameters["device"]), requires_grad=True)
        b_pred_map = Variable((torch.ones_like(observed_series[0])*(-2.0)).to(device=parameters["device"]), requires_grad=True)
        variables["kappa"] = [a_pred_map, b_pred_map, q_pred_map]
        
    elif parameters["contrast"] == "T2":
        q_pred_map = Variable(torch.ones_like(observed_series[0]).to(device=parameters["device"]), requires_grad=True)
        a_pred_map = Variable(torch.ones_like(observed_series[0]).to(device=parameters["device"]), requires_grad=True)
        variables["kappa"] = [a_pred_map, q_pred_map]

    if parameters["model"] == "mle":
        variables["std_noise"] = Variable(torch.from_numpy(np.array(1.0)).to(device=parameters["device"]), requires_grad=True)
        variables["theta"] = Variable(torch.ones((observed_series.size()[0], 6)).to(device=parameters["device"]), requires_grad=False)

    # TODO: Get brain mask from segmented tissues
    # brain_mask = torch.zeros_like(observed_series)
    # indx = np.where(observed_series > 0.05)
    # brain_mask[indx] = 1
    # brain_mask = torch.from_numpy(brain_mask)

    brain_mask = []
    
    return variables, brain_mask


def save_hdf5(data, path, filename):
    hf = h5py.File(os.path.join(path, filename), 'w')
    data = data
    for key, value in data.items():
        g = hf.create_group(key)
        if not key =='mask':
            for key_name, value_name in value.items():
                g.create_dataset(key_name, data=value_name)
        else:
            pass
            g.create_dataset(key, data=value)

    hf.close()


def load_hdf5(path):
    hf = h5py.File(path, 'r', swmr=False)

    dataset1 = hf.get('gt_maps')
    dataset2 = hf.get('weighted_images')
    dataset3 = hf.get('mask')

    return dataset1, dataset2, dataset3


def compare_predictions(a, b):
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(a, vmin=0.0, vmax=3.5)
    ax[1].imshow(b, vmin=0.0, vmax=3.5)
    ax[2].imshow(a - b, vmin=-0.2, vmax=0.2)
    ax[3].plot(a, a - b, 'x')
    ax[3].plot([0,5], [0,0], '-k', linewidth=0.5)
    plt.show()