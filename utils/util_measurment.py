import torch
import torch.nn.functional as F
import math
import numpy as np
import scipy.io as sio

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).float()
    return mask3d_batch

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def generate_shift_masks(mask_path = "/home/wyz/pythonproject/DiffBIR copy/mask", batch_size = 1):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch

def init_mask(mask_path = "/home/wyz/pythonproject/DiffBIR copy/mask", mask_type = "Phi_PhiPhiT", batch_size = 1, device = None):
    mask3d_batch = generate_masks(mask_path, batch_size)
    input_mask = None
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
        input_mask = input_mask.to(device)
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        Phi_batch = Phi_batch.to(device)
        Phi_s_batch = Phi_s_batch.to(device)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
        input_mask = input_mask.to(device)
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_meas(gt, mask, input_setting = "H"):
    if input_setting == 'H':
        input_meas, meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
        return input_meas, meas
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas

def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    mask3d_batch = mask3d_batch.to(data_batch.device)
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1).to(data_batch.device)

    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H, meas
    return meas

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def init_real_mask(mask, Phi, Phi_s, mask_type):
    if mask_type == 'Phi':
        input_mask = Phi
    elif mask_type == 'Phi_PhiPhiT':
        input_mask = (Phi, Phi_s)
    elif mask_type == 'Mask':
        input_mask = mask
    elif mask_type == None:
        input_mask = None
    return input_mask

def shift_3(f, len_shift=0):
    [nC, row, col] = f.shape
    shift_f = torch.zeros(nC, row, col + (nC - 1) * len_shift).to(f.device)
    for c in range(nC):
        shift_f[c, :, c * len_shift:c * len_shift + col] = f[c, :, :]
    return shift_f