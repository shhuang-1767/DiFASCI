import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler_initial import Sampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
from models.pretrained.dauhst import DAUHST
from models.pretrained.mst import MST
from models.pretrained.padut import PADUT
from models.pretrained.dpu import DPUNet
from models.pretrained.hdnet import HDNet
from models.pretrained.ssr import SSRNet

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-r", "--ref_path", type=str, default=None, help="reference image")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length. (The number of steps that the model trained on.)")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-is", "--infer_steps", type=int, default=None, help="Diffusion length for inference")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["cave", "ntire", "icvl", "harvard"], default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, choices=["mst", "dauhst", "padut", "admm", "lambda", "ssr", "dpu", "hdnet"], default=None, help="Select the pretrained model from: model1, model2, model3")
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="SinSR",
            choices=['realsrx4', 'bicsrx4_opencv', 'bicsrx4_matlab'],
            help="Chopping forward.",
            )
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--gpu", type=str, default=None)
    
    args = parser.parse_args()
    if args.infer_steps is None:
        args.infer_steps = args.steps
    print(f"[INFO] Using the inference step: {args.steps}")
    return args

def get_configs(args):
    if args.config is None:
        if args.task == "SinSR":
            configs = OmegaConf.load('./configs/DiFA.yaml')
        elif args.task == 'realsrx4':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    ckpt_path = args.ckpt
    
    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def main():
    args = get_parser()


    configs, chop_stride = get_configs(args)
    device = args.gpu
    configs.train.gpu = device

    resshift_sampler = Sampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=True,
            seed=args.seed,
            ddim=args.ddim
            )
    if(args.pretrained_model == "dauhst"):
       model = DAUHST(num_iterations=9, pretrained_model_path=args.ckpt).to(device)
    if(args.pretrained_model == "padut"):
       model = PADUT(in_c=28,  n_feat=28,  nums_stages=2, n_depth=3, pretrained_model_path=args.ckpt).to(device)
    if(args.pretrained_model == "mst"):
       model = MST(dim=28, stage=2, num_blocks=[4,7,5], pretrained_model_path=args.ckpt).to(device)
    if(args.pretrained_model == "ssr"):
       model = SSRNet(stage=9, pretrained_model_path=args.ckpt).to(device)
    if(args.pretrained_model == "dpu"):
       model = DPUNet(stage=9, pretrained_model_path=args.ckpt).to(device)
    if(args.pretrained_model == "hdnet"):
       model = HDNet(pretrained_model_path=args.ckpt).to(device)
    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=False, model=model, pretrained_model=args.pretrained_model, dataset=args.dataset, device=device)
        
if __name__ == '__main__':
    main()