import os, sys, math, random
from torchvision.utils import save_image
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh
from utils.util_image import *
from utils.util_measurment import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from thop import profile
torch.cuda.device_count()
torch.empty(2, device='cuda')

import lpips
from PIL import Image
import torch
import torchvision.transforms as transforms

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            desired_min_size=64,
            seed=11000,
            ddim=False
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        self.ddim=ddim
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        #assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.getenv("LOCAL_RANK", "0"))


    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).to(self.configs.train.gpu)
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        #state = torch.load(ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch>=2.0 支持 weights_only
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

class Sampler(BaseSampler):    
    def sample_func(self, lq, meas, noise_repeat=False, one_step=False, apply_decoder=True, device=None):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''

        if noise_repeat:
            self.setup_seed()

        desired_min_size = self.desired_min_size
        ori_h, ori_w = lq.shape[2:]

        model_kwargs={'lq':lq} if self.configs.model.params.cond_lq else None
        
        if not self.ddim:        
            results = self.base_diffusion.p_sample_loop(
                    y=lq,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=noise_repeat,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        else:
            results = self.base_diffusion.ddim_sample_loop(
                    y=lq,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=True,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        if not apply_decoder:
            return results["pred_xstart"]
        return results.clamp_(-1.0, 1.0)

    
    def inference(self, in_path, out_path, bs=1, noise_repeat=False, one_step=False, return_tensor=False, apply_decoder=False, model=None, dataset=None, pretrained_model=None, device=None):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(lq, meas):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''
            im_sr_tensor = self.sample_func(lq,
                    meas,
                    noise_repeat=noise_repeat, one_step=one_step, apply_decoder=False
                    )     # 1 x c x h x w, [-1, 1]
            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        return_res = {}
        import re
        im_path_list = sorted([x for x in in_path.glob("*.mat")], key=lambda x: int(re.search(r'(\d+)', x.name).group()))
        self.write_log(f'Find {len(im_path_list)} images in {in_path}')
        batch_size = 10
        results_gt = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
        results_lq = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
        results_residual = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
        results_inference = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
        i = 0
        ssim_values = []
        psnr_values = []
        for im_path in im_path_list:
            if(dataset == "ntire"):
               gt = LoadNTIRETesting(im_path).unsqueeze(0).to(device)
            elif(dataset == "icvl"):
               gt = LoadICVLTesting(im_path).unsqueeze(0).to(device)
            elif(dataset == "harvard"):
               gt = LoadHarvardTesting(im_path).unsqueeze(0).to(device)
            import time
            start_time = time.time()
            if(pretrained_model == "padut" or pretrained_model == "dauhst"):
               mask3d_batch_train, input_train_mask = init_mask(batch_size=gt.shape[0], mask_type="Phi_PhiPhiT", device = gt.device)
               mask3d_batch_train = mask3d_batch_train.to(device)
               meas = init_meas(gt, mask3d_batch_train, "Y").to(device)
               im_lq = model(meas, input_train_mask)
               residual = (_process_per_image(im_lq, meas.to(device)) + 1) / 2
            if(pretrained_model == "hdnet"):
               mask3d_batch_train, input_train_mask = init_mask(batch_size=gt.shape[0], mask_type="Phi_PhiPhiT")
               mask3d_batch_train = mask3d_batch_train.to(device)
               input_meas, meas = init_meas(gt, mask3d_batch_train, "H")
               input_meas = input_meas.to(device)
               im_lq = model(input_meas)
               residual = (_process_per_image(im_lq, meas.to(device)) + 1) / 2
            elif(pretrained_model == "mst"):
               mask3d_batch_train, input_train_mask = init_mask(batch_size=gt.shape[0], mask_type="Phi", device = gt.device)
               mask3d_batch_train = mask3d_batch_train.to(device)
               input_train_mask = input_train_mask.to(device)
               meas = init_meas(gt, mask3d_batch_train, "H")
               input_meas = meas[0].to(device)
               im_lq = model(input_meas, input_train_mask)
               residual = (_process_per_image(im_lq, meas[1].to(device)) + 1) / 2
            elif(pretrained_model == "ssr" or pretrained_model == "dpu"):
                mask_path = "mask/mask_256_28.mat"
                mask = sio.loadmat(mask_path)['mask']
                mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(device).float()
                Phi_batch = mask.unsqueeze(0).repeat(gt.shape[0], 1, 1, 1)
                Phi_s_batch = torch.zeros((gt.shape[0], 256, 310), dtype=gt.dtype).to(device)
                for k in range(gt.shape[0]):
                  Phi_s_batch[k] = torch.sum(shift_3(Phi_batch[k], 2) ** 2, 0)
                Phi_s_batch[Phi_s_batch == 0] = 1
                meas = torch.zeros((gt.shape[0], 256, 310), dtype=gt.dtype).to(device)
                for k in range(gt.shape[0]):
                   mea = shift_3(mask * gt[k], len_shift= 2)
                   meas[k] = torch.sum(mea, 0).float()
                lq = model(meas, input_mask = (Phi_batch, Phi_s_batch))
                lq = lq[8].clamp(min=0., max=1.)
                im_lq = lq
                residual = (_process_per_image(im_lq, meas.to(device)) + 1) / 2
            im_hq = im_lq + residual
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"run time: {elapsed_time} s")
            residual_save = (im_lq - im_lq.min()) / (im_lq.max() - im_lq.min()) - (im_hq - im_hq.min()) / (im_hq.max() - im_hq.min())
            results_gt[i] = np.array(gt.cpu().permute(0, 2, 3, 1).numpy(), copy = True)
            results_residual[i] = np.array((residual_save.detach().cpu().permute(0, 2, 3, 1)).numpy())
            results_lq[i] = np.array(im_lq.detach().cpu().permute(0, 2, 3, 1))
            results_inference[i] = np.array((im_hq.detach().cpu().permute(0, 2, 3, 1)).numpy())
            psnr_value = psnr(results_gt[i], results_inference[i], data_range=1.0)
            psnr_values.append(psnr_value)
            print(f"PSNR: {psnr_value} dB")
            ssim_value = ssim(results_gt[i], results_inference[i], data_range=1.0)
            print(f"SSIM: {ssim_value} dB")
            ssim_values.append(ssim_value)
            i = i + 1       
        sio.savemat("results/initial_predictor.mat", {'truth':  results_gt, 'pred':  results_lq})
        sio.savemat("results/one_step_diffusion.mat", {'truth':  results_gt, 'pred': results_residual})
        sio.savemat("results/refined.mat",  {'truth':  results_gt, 'pred':  results_inference})
        average_psnr = np.mean(np.array(psnr_values))
        print(f"Average_PSNR: {average_psnr} dB")
        average_ssim = np.mean(np.array(ssim_values))
        print(f"Average_SSIM: {average_ssim}")
        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
        return return_res
    
if __name__ == '__main__':
    pass