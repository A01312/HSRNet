from prettytable import PrettyTable
import os
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", choices=['best.pth', 'last.pth'], default='last.pth')
    cli = parser.parse_args()
    args = load_train_configs(cli.config_file)
    args.output_dir = op.dirname(op.abspath(cli.config_file))
    args.training = False
    logger = setup_logger('HSRNet', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    state = torch.load(op.join(args.output_dir, cli.checkpoint), map_location='cpu')['model']
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
    do_inference(model.cuda(), test_img_loader, test_txt_loader)
