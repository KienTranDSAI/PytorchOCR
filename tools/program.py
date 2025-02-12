# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import sys
import platform
import yaml
import time
import datetime
# import paddle
# import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import torch
import platform

# from ppocr.utils.stats import TrainingStats
# from ppocr.utils.save_load import save_model
# from ppocr.utils.utility import print_dict, AverageMeter
# from pytorchocr.utils.logging import get_logger
# from pytorchocr.utils.loggers import WandbLogger, Loggers
# from ppocr.utils import profiler
# from ppocr.data import build_dataloader
# from ppocr.utils.export_model import export


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "-p",
            "--profiler_options",
            type=str,
            default=None,
            help="The option of profiler, which should be in format "
            '"key1=value1;key2=value2;key3=value3".',
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split(".")
            assert sub_keys[0] in config, (
                "the sub_keys can only be one of global_config: {}, but get: "
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_gcu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = (
        "Config {} cannot be set as true while your paddle "
        "is not compiled with {} ! \nPlease try: \n"
        "\t1. Install paddlepaddle to run model on {} \n"
        "\t2. Set {} as false in config file to run "
        "model on CPU"
    )

    pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], torch.Tensor):
                preds[k] = preds[k].to(dtype=torch.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], torch.Tensor):
                preds[k] = preds[k].to(dtype=torch.float32)
    elif isinstance(preds, torch.Tensor):
        preds = preds.to(dtype=torch.float32)
    return preds

def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = torch.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] + feat[idx_time]
                    ) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc="get center:")
    max_iter = (
        len(eval_dataloader) - 1
        if platform.system() == "Windows"
        else len(eval_dataloader)
    )
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        # update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    if is_train:
        # save_config
        save_model_dir = config["Global"]["save_model_dir"]
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
            yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = "{}/train.log".format(save_model_dir)
    else:
        log_file = None
    use_gpu = config['Global']['use_gpu']
    alg = config["Architecture"]["algorithm"]
    return config

def train(
    config,
    train_dataloader,
    valid_dataloader,
    device,
    model,
    loss_class,
    optimizer,
    lr_scheduler,
    post_process_class,
    eval_class,
):
    start_epoch = 0
    epoch_num = config["Global"]["epoch_num"]
    eval_batch_step = config["Global"]["eval_batch_step"]
    eval_batch_epoch = config["Global"].get("eval_batch_epoch", None)

    print_batch_step = config["Global"]["print_batch_step"]
    global_step = 0
    start_eval_step = 0
    if isinstance(eval_batch_step, list) and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0] if not eval_batch_epoch else 0
        eval_batch_step = (
            eval_batch_step[1]
            if not eval_batch_epoch
            else step_pre_epoch * eval_batch_epoch
        )
    
    for epoch in range(start_epoch, epoch_num + 1):
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Progress"):
            model.to(device)
            batch = [item.to(device) for item in batch]  
            model.train()
            preds = model(batch[0])
            preds = to_float32(preds)
            
            loss = loss_class(preds, batch)
            avg_loss = loss["loss"]
            
            avg_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

        if global_step > start_eval_step and (global_step - start_eval_step) % eval_batch_step == 0:
            cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    device)
            cur_metric_str = "cur metric, {}".format(
                    ", ".join(["{}: {}".format(k, v) for k, v in cur_metric.items()])
                )
            print(cur_metric_str)
            
def eval(model,
        valid_dataloader,
        post_process_class,
        eval_class,
        device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader), desc="eval model:", position=0, leave=True
        )
        max_iter = (
            len(valid_dataloader) - 1
            if platform.system() == "Windows"
            else len(valid_dataloader)
        )
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            batch = [item.to(device) for item in batch] 
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()
            preds = model(images)
            batch_numpy = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_numpy.append(item.cpu().numpy())
                else:
                    batch_numpy.append(item)
            total_time += time.time() - start
            post_result = post_process_class(preds, batch_numpy[1])
            eval_class(post_result, batch_numpy)
            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        metric = eval_class.get_metric()
    pbar.close()
    model.train()
    if total_time > 0:
        metric["fps"] = total_frame / total_time
    else:
        metric["fps"] = 0
    return metric