from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import nibabel as nib
from scipy.io import loadmat
import SimpleITK as sitk

## Image loading
def load_hdf5(path):
    hf = h5py.File(path, 'r', swmr=False)

    dataset1 = hf.get('gt_maps')
    dataset2 = hf.get('weighted_images')
    dataset3 = hf.get('mask')

    return dataset1, dataset2, dataset3


def loadData(path):
    data = {}
    h = load_hdf5(path)
    data["weighted_series"] = np.asarray(h[1]["weighted_series"], dtype=np.float32)
    data["echo_times"] = np.asarray(h[1]["echo_times"], dtype=np.float32)
    data["q_star_map"] = np.asarray(h[0]["q_star_map"], dtype=np.float32)
    data["q_map"] = np.asarray(h[0]["q_map"], dtype=np.float32)
    data["pd_map"] = np.asarray(h[0]["ro_map"], dtype=np.float32)
    data["mask"] = np.asarray(h[2]["mask"], dtype=np.float32)
    
    return data


def convertToNii(data):
    nii_im = nib.Nifti1Image(data, affine=np.eye(4))
    return nii_im


def loadSyntheticData(filepath, sliceSelect=[]):
    print("Loading data")
    origData = loadData(filepath)
    data_flipped = np.flip(origData["weighted_series"],[1,2,3])
    
    if sliceSelect:
        dataW = data_flipped[:, :, :, sliceSelect]
        groundTruthMaps = np.flip(origData['q_star_map'])[:, :, sliceSelect]
        mask = np.flip(origData['mask'])[:,:,sliceSelect]
    else:
        dataW = data_flipped
        groundTruthMaps = np.flip(origData['q_star_map'])
        mask = np.flip(origData['mask'])
        
    observedSeries = {}
    observedSeries['weighted_series'] = np.array(dataW).copy()
    observedSeries['echo_times'] = origData['echo_times']
    print("Data loaded")

    return observedSeries, groundTruthMaps, mask


def loadRealData(filepath, sliceSelect=[]):
    print("Loading data")
    matFile = loadmat(filepath)
    
    if sliceSelect:
        data = matFile["IR4D_ord"].astype(float)[...,sliceSelect,:]
    else:
        data = matFile["IR4D_ord"].astype(float)
        
    dataNorm = data/1000
    
    observedSeries = {}
    observedSeries['weighted_series'] = np.transpose(dataNorm, (2, 0, 1))
    observedSeries['echo_times'] = matFile["TI"][0]/1000
    
    print("Data loaded")
    return observedSeries


def prepareDataForElastix(filepath):
    sitk_data_fixed = sitk.ReadImage(filepath)
    sitk_data_moving = sitk.ReadImage(filepath)
    
    return sitk_data_fixed, sitk_data_moving


def saveNii(data, filepath):
    print("saving data")
    data = np.transpose(data, (1,2,0))
    data_nii = convertToNii(data)
    niiDataPath = filepath
    
    nib.save(data_nii, niiDataPath)
    print("save finished")


def readNii(filepath, sliceSelect=[]):
    data = nib.load(filepath)
    data = data.get_data()
    data = np.transpose(data, (2,0,1))
    
    if sliceSelect:
        data = data[:,...,sliceSelect]
    else:
        data = data
    return data