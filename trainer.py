import os, sys, math, time, random, datetime, functools
#import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
import copy
from datapipe.datasets import create_dataset

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image
from utils.util_measurment import * 

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import scipy.io as sio
from peft import LoraConfig, get_peft_model, inject_adapter_in_model

import torch
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, CLIPTextModel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

max_psnr = 0

def Quality_Index(ref, img, logger):
    N = ref.shape[0]
    psnrs = torch.zeros(N, dtype=torch.float32)
    ssims = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        psnrs[i] = psnr(ref[i].unsqueeze(0).cpu().numpy(), img[i].unsqueeze(0).cpu().numpy(), data_range=1.0)
        ssims[i] = torch.tensor(ssim(ref[i].unsqueeze(0).cpu().numpy(), img[i].unsqueeze(0).cpu().numpy(), data_range=1.0, channel_axis=0))
        logger.info(f'Sceen {i}: PSNR={psnrs[i].item()}, SSIM={ssims[i].item()}')
    return psnrs.mean(), ssims.mean()

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = 1

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = 0
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = 0
    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        # only should be run on rank: 0
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
        else:
            save_dir = Path(self.configs.save_dir) / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # text logging
        if self.rank == 0:
            logtxet_path = save_dir / 'training.log'
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a')
            self.logger.add(sys.stdout, format="{message}", level="INFO")

        # tensorboard logging
        if self.rank == 0:
            # log_dir = save_dir / 'tf_logs'
            # if not log_dir.exists():
                # log_dir.mkdir()
            # self.writer = SummaryWriter(str(log_dir))
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # image saving
        if self.rank == 0 and self.configs.train.save_images:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # checkpoint saving
        if self.rank == 0:
            ckpt_dir = save_dir / 'ckpts'
            if not ckpt_dir.exists():
                ckpt_dir.mkdir()
            self.ckpt_dir = ckpt_dir

        # ema checkpoint saving
        if self.rank == 0 and hasattr(self, 'ema_rate'):
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            if not ema_ckpt_dir.exists():
                ema_ckpt_dir.mkdir()
            self.ema_ckpt_dir = ema_ckpt_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0:
            # self.writer.close()
            pass

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)


        if self.configs.train.resume:
            assert self.configs.train.resume.endswith(".pth") and os.path.isfile(self.configs.train.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.train.resume}")
            ckpt = torch.load(self.configs.train.resume, map_location=fdevice)
            util_net.reload_model(self.model, ckpt['state_dict'])
            util_net.reload_model(self.initial_predictor, ckpt['pretrained'])

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(self.iters_start):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                #ema_ckpt_path = self.ema_ckpt_dir / (Path(self.configs.train.ema_resume).name)
                ema_ckpt_path = self.configs.train.ema_resume
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=fdevice)
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.initial_predictor.parameters()),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            self.model = DDP(model.to(device), device_ids=[self.rank,], broadcast_buffers=False)  # wrap the network
        else:
            self.model = model.to(device)

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=False,
                        num_workers=self.configs.train.get('num_workers', 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info("Detailed network architecture:")
            self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")
            
    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.to(device).to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def build_iqa(self):
        import pyiqa
        if self.rank == 0:
            self.metric_dict={}
            self.metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
            self.metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
        
    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler
        
        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)
            
            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_sheduler')
        self.lr_sheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            torch.save({'iters_start': self.current_iters,
                        'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                        'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                        'state_dict': self.model.state_dict(),
                        'pretrained': self.initial_predictor.state_dict()}, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, num_timesteps //2, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.2e}/{:.2e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            )
                    # tensorboard
                    # self.writer.add_scalar(f'Loss-Step-{current_record}',
                                           # self.loss_mean['loss'][jj].item(),
                                           # self.log_step[phase])
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.log_step[phase] += 1
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                x1 = vutils.make_grid(batch['lq'], normalize=True, scale_each=True)  # c x h x w
                # self.writer.add_image("Training LQ Image", x1, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x1.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"lq_{self.log_step_img[phase]:05d}.png",
                           )
                x2 = vutils.make_grid(batch['gt'], normalize=True)
                # self.writer.add_image("Training HQ Image", x2, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x2.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"hq_{self.log_step_img[phase]:05d}.png",
                           )
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                x3 = vutils.make_grid(x_t, normalize=True, scale_each=True)
                # self.writer.add_image("Training Diffused Image", x3, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x3.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"diffused_{self.log_step_img[phase]:05d}.png",
                           )
                x0_pred = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z0_pred, tt),
                        self.autoencoder,
                        )
                x4 = vutils.make_grid(x0_pred, normalize=True, scale_each=True)
                # self.writer.add_image("Training Predicted Image", x4, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x4.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"x0_pred_{self.log_step_img[phase]:05d}.png",
                           )
                self.log_step_img[phase] += 1

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)
                
class TrainerDifIR(TrainerBase):
    def __init__(self, configs):
        # ema settings
        self.ema_rate = configs.train.ema_rate
        super().__init__(configs)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            self.model = DDP(model.to(device), device_ids=[self.rank,], broadcast_buffers=False)  # wrap the network
        else:
            self.model = model.to(device)
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=fdevice)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model, ckpt)

        # EMA
        if self.rank == 0:
            self.ema_model = deepcopy(model).to(device)
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half().to(device)
            else:
                self.autoencoder = autoencoder.to(device)
        else:
            self.autoencoder = None

        # LPIPS metric
        if self.rank == 0:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

        # model information
        self.print_model_info()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(device)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).to(device)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        input_setting = self.configs.pretrained.input_setting
        input_mask = self.configs.pretrained.input_mask
        gt = data.to(device)
        mask3d_batch_train, input_train_mask = init_mask(mask_path=self.configs.mask_path, batch_size=data.shape[0], mask_type=input_mask, device = device)
        if(input_setting == "Y"):
            meas = init_meas(gt, mask3d_batch_train, input_setting).to(device)
            lq = self.initial_predictor(meas, input_train_mask)
        if(input_setting == "H"): 
            if(self.configs.pretrained.name == 'hdnet'):
               input_meas, meas = init_meas(gt, mask3d_batch_train, input_setting)
               input_meas = input_meas.to(device)
               meas = meas.to(device)
               lq = self.initial_predictor(input_meas)
            else:
               input_train_mask = input_train_mask.to(device)
               input_meas, meas = init_meas(gt, mask3d_batch_train, input_setting)
               input_meas = input_meas.to(device)
               meas = meas.to(device)
               lq = self.initial_predictor(input_meas, input_train_mask)
        if(input_setting == "SSR" or input_setting == "DPU"):
            mask_path = "mask/mask_256_28.mat"
            mask = sio.loadmat(mask_path)['mask']
            mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(device).float()
            Phi_batch = mask.unsqueeze(0).repeat(data.shape[0], 1, 1, 1)
            Phi_s_batch = torch.zeros((data.shape[0], 256, 310), dtype=gt.dtype).to(device)
            for i in range(data.shape[0]):
              Phi_s_batch[i] = torch.sum(shift_3(Phi_batch[i], 2) ** 2, 0)
            Phi_s_batch[Phi_s_batch == 0] = 1
            meas = torch.zeros((data.shape[0], 256, 310), dtype=gt.dtype).to(device)
            for i in range(data.shape[0]):
               mea = shift_3(mask * gt[i], len_shift= 2)
               meas[i] = torch.sum(mea, 0).float()
            lq = self.initial_predictor(meas, input_mask = (Phi_batch, Phi_s_batch))
            lq = lq[8].clamp(min=0., max=1.)
        return {'lq':lq, 'gt':gt, 'meas':meas}


    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"{device}",
                    )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1) if self.configs.autoencoder is not None else 1
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            noise = torch.randn(
                    size=micro_data['gt'].shape[:2] + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            model_kwargs={'lq':micro_data['lq'],} if self.configs.model.params.cond_lq else None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses, z_t, z0_pred = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses, z_t, z0_pred = compute_losses()
                    loss = losses["loss"].mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses, z_t, z0_pred = compute_losses()
                else:
                    with self.model.no_sync():
                        losses, z_t, z0_pred = compute_losses()
                loss = losses["loss"].mean() / num_grad_accumulate
                loss.backward()

            # make logging
            self.log_step_train(losses, tt, micro_data, z_t, z0_pred, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        if len(self.configs.train.milestones) > 0:
            base_lr = self.configs.train.lr
            linear_steps = self.configs.train.milestones[0]
            current_iters = self.current_iters if current_iters is None else current_iters
            if current_iters <= linear_steps:
                for params_group in self.optimizer.param_groups:
                    params_group['lr'] = (current_iters / linear_steps) * base_lr
            elif current_iters in self.configs.train.milestones:
                for params_group in self.optimizer.param_groups:
                    params_group['lr'] *= 0.5
        else:
            pass


    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = [int(self.base_diffusion.num_timesteps * x) for x in [0.25, 0.5, 0.75, 1]]
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = mean_musiq = mean_clipiqa = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']
                num_iters = 0
                model_kwargs={'lq':im_lq,} if self.configs.model.params.cond_lq else None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).to(device)
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        ):
                    sample_decode = {}
                    if (num_iters + 1) in indices or num_iters + 1 == 1:
                        for key, value in sample.items():
                            if key in ['sample', 'pred_xstart']:
                            # if key in ['sample']:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        self.base_diffusion._scale_input(value, tt-1), # 难道这里要改
                                        self.autoencoder,
                                        )
                        im_sr_progress = sample_decode['sample']
                        im_xstart = sample_decode['pred_xstart']
                        if num_iters + 1 == 1:
                            im_sr_all, im_xstart_all = im_sr_progress, im_xstart
                            # im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                            im_xstart_all = torch.cat((im_xstart_all, im_xstart), dim=1)
                    num_iters += 1
                    tt -= 1

                with torch.no_grad():
                    results = sample_decode['sample'].detach()
                    mean_clipiqa += self.metric_dict["clipiqa"](results.detach() * 0.5 + 0.5).sum().item()
                    mean_musiq += self.metric_dict["musiq"](results.detach() * 0.5 + 0.5).sum().item()
                    
                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'].detach() * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=True,
                            )
                    mean_lpips += self.lpips_loss(sample_decode['sample'].detach(), im_gt).sum().item()
                    
                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    im_xstart_all = rearrange(im_xstart_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    x1 = vutils.make_grid(im_sr_all.detach(), nrow=len(indices)+1, normalize=True, scale_each=True)
                    x2 = vutils.make_grid(im_xstart_all.detach(), nrow=len(indices)+1, normalize=True, scale_each=True)
                    # self.writer.add_image('Validation Sample Progress', x1, self.log_step_img[phase])
                    if self.configs.train.save_images:
                        util_image.imwrite(
                               x1.cpu().permute(1,2,0).numpy(),
                               self.image_dir / phase / f"progress_{self.log_step_img[phase]:05d}.png",
                               )
                        util_image.imwrite(
                               x2.cpu().permute(1,2,0).numpy(),
                               self.image_dir / phase / f"predict_x_{self.log_step_img[phase]:05d}.png",
                               )
                    x3 = vutils.make_grid(im_lq, normalize=True)
                    # self.writer.add_image('Validation LQ Image', x3, self.log_step_img[phase])
                    if self.configs.train.save_images:
                        util_image.imwrite(
                               x3.cpu().permute(1,2,0).numpy(),
                               self.image_dir / phase / f"lq_{self.log_step_img[phase]:05d}.png",
                               )
                    if 'gt' in data:
                        x4 = vutils.make_grid(im_gt, normalize=True)
                        # self.writer.add_image('Validation HQ Image', x4, self.log_step_img[phase])
                        if self.configs.train.save_images:
                            util_image.imwrite(
                                   x4.cpu().permute(1,2,0).numpy(),
                                   self.image_dir / phase / f"hq_{self.log_step_img[phase]:05d}.png",
                                   )
                    self.log_step_img[phase] += 1

            mean_clipiqa /= len(self.datasets[phase])
            mean_musiq /= len(self.datasets[phase])
            self.logger.info(f'Validation Metric: MUSIQ={mean_musiq:5.2f}, clipiqa={mean_clipiqa:6.4f}...')
            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                self.log_step[phase] += 1

            self.logger.info("="*100)

            if not self.configs.train.use_ema_val:
                self.model.train()

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if not 'relative_position_index' in key:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

class TrainerDistillDifIR(TrainerDifIR):
    def __init__(self, configs):
        super().__init__(configs)
        self.distill_ddpm = configs.train.get("distill_ddpm", False)
        self.uncertainty_hyper = configs.train.get("uncertainty_hyper", False)
        self.uncertainty_num_aux = configs.train.get("uncertainty_num_aux", 2)
        self.use_reflow = configs.train.get("use_reflow", False)
        self.learn_xT = configs.train.get("learn_xT", False)
        self.reformulated_reflow = configs.train.get("reformulated_reflow", False)
        self.finetune_use_gt = configs.train.get("finetune_use_gt", False)
        self.xT_cov_loss = configs.train.get("xT_cov_loss", False)
        self.loss_in_image_space = configs.train.get("loss_in_image_space", False)
        
    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)
    

    def build_model(self):
        global device
        gpu_id = self.configs.train.gpu
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"[INFO]: gpu_id {device}")

        params = self.configs.model.get('params', dict)
        params_teacher = self.configs.teacher_model.get("params", dict)
        
        heterogeneous_model = False
        if params_teacher is None: params_teacher = params
        else: heterogeneous_model = True

        model = util_common.get_obj_from_str(self.configs.model.target)(**params)                
        teacher_model = util_common.get_obj_from_str(self.configs.teacher_model.target)(**params_teacher)

        self.teacher_model = teacher_model.to(device)
            
        teacher_ckpt_path = self.configs.teacher_model.teacher_ckpt_path
        if self.rank == 0:
            self.logger.info(f"[INFO]: Initializing the teacher model from {teacher_ckpt_path}")
        ckpt = torch.load(teacher_ckpt_path, map_location = device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(self.teacher_model, ckpt) 

        self.model = model.to(device)

        # pretrained initial_predictor
        self.initial_predictor = util_common.instantiate_from_config(self.configs.pretrained).to(device)
        
        # EMA
        if self.rank == 0:
            self.ema_model = deepcopy(model if not heterogeneous_model else model).to(device)
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half().to(device)
            else:
                self.autoencoder = autoencoder.to(device)
        else:
            self.autoencoder = None

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

        self.print_model_info()

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        losses_dict = {}
        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=device,
                    )
            
            if not self.use_reflow:
                tt = torch.ones_like(tt) * (self.base_diffusion.num_timesteps - 1) # fix the time step of the student model

            loss_type = self.configs.train.loss
            model_kwargs={'lq':micro_data['lq'],} if self.configs.model.params.cond_lq else None
            input_setting = self.configs.pretrained.input_setting
            input_mask = self.configs.pretrained.input_mask
            model = self.configs.pretrained.name
            
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses, z_t, z0_pred = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses, z_t, z0_pred = compute_losses()
                    loss = losses["loss"].mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    if(loss_type == "DISTILL"):
                       compute_losses = functools.partial(
                           self.base_diffusion.training_losses_distill,
                           self.model,
                           self.teacher_model,
                           self.autoencoder,
                           None,
                           micro_data['lq'],
                           data['meas'],
                           input_setting,
                           input_mask,
                           "MC",
                           tt,
                           device = device,
                           model_kwargs=model_kwargs,
                           noise=None,
                       )
                       losses, z_t, x_1 = compute_losses()
                       losses_dict["MC"] = losses["MC"]
                       # group opration on x_1
                       x_2 = torch.rot90(x_1, k=1, dims=(2, 3))
                       mask3d_batch_train, input_train_mask = init_mask(batch_size=x_2.shape[0], mask_type=input_mask, device=x_1.device)
                       mask3d_batch_train = mask3d_batch_train.to(device)
                       if(input_setting == "Y"):
                           y_2 = init_meas(x_2, mask3d_batch_train, input_setting).to(device)
                           re_x_init = self.initial_predictor(y_2, input_train_mask).to(device)
                       if(input_setting == "H"):
                        if(model == 'hdnet'):
                           y_2, _ = init_meas(x_2, mask3d_batch_train, input_setting)
                           y_2 = y_2.to(device)
                           re_x_init = self.initial_predictor(y_2).to(device) 
                        else:
                           input_train_mask = input_train_mask.to(device)
                           y_2, _ = init_meas(x_2, mask3d_batch_train, input_setting)
                           y_2 = y_2.to(device)
                           re_x_init = self.initial_predictor(y_2, input_train_mask).to(device)
                       if(input_setting == "SSR" or input_setting == "DPU"):
                           mask_path = "mask/mask_256_28.mat"
                           mask = sio.loadmat(mask_path)['mask']
                           mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(device).float()
                           Phi_batch = mask.unsqueeze(0).to(device)
                           Phi_s_batch = torch.sum(shift_3(mask, 2) ** 2, 0).unsqueeze(0).to(device)
                           Phi_s_batch[Phi_s_batch == 0] = 1
                           y_2 = shift_3(mask * x_2[0], len_shift= 2)
                           y_2 = torch.sum(y_2, 0).unsqueeze(0).to(device)
                           re_x_init = self.initial_predictor(y_2, input_mask = (Phi_batch, Phi_s_batch))[8].clamp(min=0., max=1.)
                       compute_losses_ei = functools.partial(
                            self.base_diffusion.training_losses_distill,
                            self.model,
                            self.teacher_model,
                            self.autoencoder,
                            x_2, # roated x_1
                            re_x_init, # re-reconstructed x_init
                            data['meas'],
                            input_setting,
                            input_mask,
                            "EC",
                            tt,
                            device,
                            model_kwargs=model_kwargs,
                            noise=None
                        )
                       losses, z_t, z0_pred = compute_losses_ei()
                       losses_dict["EC"] = losses["EC"]

                       compute_losses_distill = functools.partial(
                            self.base_diffusion.training_losses_distill,
                            self.model,
                            self.teacher_model,
                            self.autoencoder,
                            x_1, # 
                            micro_data['lq'],
                            data['meas'],
                            input_setting,
                            input_mask,
                            "DISTILL",
                            tt,
                            device,
                            model_kwargs=model_kwargs,
                            noise=None
                        )
                       losses, z_t, z0_pred = compute_losses_distill()
                       losses_dict["DISTILL"] = losses["DISTILL"]
                       losses_dict["loss"] = losses_dict["MC"] + losses_dict["EC"] + 0.001 * losses_dict["DISTILL"]
                else:
                    with self.model.no_sync():
                        losses, z_t, z0_pred = compute_losses()
                loss = losses_dict["loss"].mean() / num_grad_accumulate
                loss.backward()

            # make logging
            self.log_step_train(losses_dict, tt*0 if not self.use_reflow else tt, micro_data, z_t, z0_pred, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()
        
        
    def log_step_train(self, loss, tt, batch, z_t, z0_pred, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, num_timesteps //2, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt)).to(device)
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if (self.current_iters % self.configs.train.log_freq[0] == 0 or self.current_iters == 1) and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                    
                log_str = 'Train: {:06d}/{:06d}: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                
                for key, val in self.loss_mean.items():
                    log_str += f'{key}:{val[0].item():.2e} '
            
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.log_step[phase] += 1
                
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                x1 = vutils.make_grid(batch['lq'], normalize=True, scale_each=True)  # c x h x w
                # self.writer.add_image("Training LQ Image", x1, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x1.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"lq_{self.log_step_img[phase]:05d}.png",
                           )
                x2 = vutils.make_grid(batch['gt'], normalize=True)
                # self.writer.add_image("Training HQ Image", x2, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x2.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"hq_{self.log_step_img[phase]:05d}.png",
                           )
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                x3 = vutils.make_grid(x_t, normalize=True, scale_each=True)
                # self.writer.add_image("Training Diffused Image", x3, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x3.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"diffused_{self.log_step_img[phase]:05d}.png",
                           )
                x0_pred = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z0_pred, tt),
                        self.autoencoder,
                        )
                x4 = vutils.make_grid(x0_pred, normalize=True, scale_each=True)
                # self.writer.add_image("Training Predicted Image", x4, self.log_step_img[phase])
                if self.configs.train.save_images:
                    util_image.imwrite(
                           x4.cpu().permute(1,2,0).numpy(),
                           self.image_dir / phase / f"x0_pred_{self.log_step_img[phase]:05d}.png",
                           )
                self.log_step_img[phase] += 1

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    def validation(self, phase='val'):
        
        # Only evaluted the result of the first step
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = [int(self.base_diffusion.num_timesteps * x) for x in [0.25, 0.5, 0.75, 1]]
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = mean_musiq = mean_clipiqa = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt, im_meas = data['lq'], data['gt'], data['meas']
                else:
                    im_lq = data['lq']

                model_kwargs={'lq':im_lq,} if self.configs.model.params.cond_lq else None
                y=im_lq[:,:,:,0:256]
                results = self.base_diffusion.ddim_sample_loop(
                    y,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                    one_step=True,
                    phase = phase
                    )
            r = (results["sample"] + 1) / 2
            lq = data['lq'].to(device)
            pred = r + lq
            truth = data['gt'].to(device)
            global max_psnr
            if(max_psnr == 0):
                psnr, ssim = Quality_Index(truth, lq, self.logger)
                self.logger.info(f'Initial_Average:PSNR={psnr}, Initial_Average:SSIM={ssim}')
                self.logger.info("="*100)
            psnr, ssim = Quality_Index(truth, pred, self.logger)
            if(psnr > max_psnr):
                self.save_ckpt()
                max_psnr = psnr
            self.logger.info(f'Average:PSNR={psnr}, Average:SSIM={ssim}')
            self.logger.info("="*100)
            if not self.configs.train.use_ema_val:
                self.model.train()
        
def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))