import random
import numpy as np
from pathlib import Path

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset
from albumentations import SmallestMaxSize

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .degradation_bsrgan.bsrgan_light import degradation_bsrgan_variant, degradation_bsrgan
import scipy.io as sio
import h5py
from utils.util_image import *
from torch.autograd import Variable

def create_dataset(dataset_config):
    if dataset_config['type'] == 'ntire_train':
        dataset = NTIREDataTrain(dataset_config['params'])
    elif dataset_config['type'] == 'ntire_test':
        dataset = NTIREDataTest(dataset_config['params'])
    elif dataset_config['type'] == 'icvl_train':
        dataset = ICVLDataTrain(dataset_config['params'])
    elif dataset_config['type'] == 'icvl_test':
        dataset = ICVLDataTest(dataset_config['params'])
    elif dataset_config['type'] == 'harvard_train':
        dataset = HarvardDataTrain(dataset_config['params'])
    elif dataset_config['type'] == 'harvard_test':
        dataset = HarvardDataTest(dataset_config['params'])
    elif dataset_config['type'] == 'kaist_train':
        dataset = KAISTDataTrain(dataset_config['params'])
    elif dataset_config['type'] == 'kaist_test':
        dataset = KAISTDataTest(dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])
    return dataset


class KAISTDataTrain(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadCAVETesting(self.file_paths[index])
        h, w, _ = img_gt.shape
        #img_gt = shuffle_crop(img_gt)
        out = torch.from_numpy(img_gt).permute(2, 0, 1)
        return out
        
class KAISTDataTest(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadCAVETesting(self.file_paths[index])
        h, w, _ = img_gt.shape
        out = torch.from_numpy(img_gt)
        return out.permute(2, 0, 1)

    
class NTIREDataTrain(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadNTIRETraining(self.file_paths[index])
        h, w, _ = img_gt.shape
        img_gt = shuffle_crop(img_gt)
        out = img_gt
        return out
    
class NTIREDataTest(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadNTIRETesting(self.file_paths[index])
        h, w, _ = img_gt.shape
        out = img_gt
        return out

class ICVLDataTrain(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadICVLTraining(self.file_paths[index])
        h, w, _ = img_gt.shape
        img_gt = shuffle_crop(img_gt)
        out = img_gt
        return out.float()
    
class ICVLDataTest(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadICVLTesting(self.file_paths[index])
        h, w, _ = img_gt.shape
        out = img_gt
        return out.float()
    
class HarvardDataTrain(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadHarvardTraining(self.file_paths[index])
        h, w, _ = img_gt.shape
        img_gt = shuffle_crop(img_gt)
        #img_gt = torch.from_numpy(np.transpose(img_gt, (2, 0 , 1)))
        out = img_gt
        return out
    
class HarvardDataTest(Dataset):
    def __init__(
            self,
            file_list,
            gt_size=256,
            im_exts=['mat']
            ):
        self.file_paths = load_file_list(file_list)
        self.gt_size = gt_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_gt = LoadHarvardTesting(self.file_paths[index])
        h, w, _ = img_gt.shape
        out = img_gt
        return out