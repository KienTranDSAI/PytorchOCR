import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from program import preprocess
from program import *
import program
from pytorchocr.data import build_dataloader, set_signal_handlers
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorchocr.postprocess import build_post_process
from pytorchocr.modeling.architectures import build_model
from pytorchocr.losses import build_loss
from pytorchocr.optimizer import build_optimizer
from pytorchocr.utils.utility import *
from pytorchocr.utils.save_load import load_model, fast_load_model
from pytorchocr.metrics import build_metric
from tqdm import tqdm

def main(config, device, seed):
    global_config = config["Global"]
    set_signal_handlers()
    

    train_dataloader = build_dataloader(config, "Train", device, seed)
    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, seed)
    step_pre_epoch = len(train_dataloader)
    post_process_class = build_post_process(config["PostProcess"], global_config)
    model = build_model(config["Architecture"])
    loss_class = build_loss(config["Loss"])
    optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            epochs=config["Global"]["epoch_num"],
            step_each_epoch=len(train_dataloader),
            model=model,
        )
    eval_class = build_metric(config["Metric"])
    
    # checkpoints = global_config.get("checkpoints")
    # pretrained_model = global_config.get("pretrained_model")
    # if checkpoints:
    #     ckt_pth = checkpoints
    # elif pretrained_model:
    #     ckt_pth = pretrained_model
    # else: 
    #     ckt_pth = None
    # if ckt_pth:
    #     fast_load_model(model, ckt_pth)
    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )
    program.train(config,
                 train_dataloader,
                 valid_dataloader,
                 device,
                 model,
                 loss_class,
                 optimizer,
                 lr_scheduler,
                 post_process_class,
                 eval_class,
                 pre_best_model_dict,
                 step_pre_epoch)
                
if __name__ == "__main__":
    config = preprocess(is_train=True)
    device = "cuda:3"
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, device, seed)