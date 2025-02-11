from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from __future__ import unicode_literals



import os

import sys

import numpy as np

import signal

import random

import copy

import torch

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler



__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))



from pytorchocr.data.imaug import transform, create_operators

from pytorchocr.data.simple_dataset import SimpleDataSet, MultiScaleDataSet



TextDetDataset = SimpleDataSet



__all__ = ["build_dataloader", "transform", "create_operators", "set_signal_handlers"]



def set_signal_handlers():

    pid = os.getpid()

    try:

        pgid = os.getpgid(pid)

    except AttributeError:

        pass

    else:

        if pid == pgid:

            signal.signal(signal.SIGINT, term_mp)

            signal.signal(signal.SIGTERM, term_mp)



def build_dataloader(config, mode, device, seed=None):

    config = copy.deepcopy(config)



    support_dict = [

        "SimpleDataSet",

        "TextDetDataset",

    ]

    

    module_name = config[mode]["dataset"]["name"]

    assert module_name in support_dict, Exception(

        "DataSet only support {}".format(support_dict)

    )

    assert mode in ["Train", "Eval", "Test"], "Mode should be Train, Eval or Test."



    # Create dataset

    dataset = eval(module_name)(config, mode, seed)

    

    # Get loader config

    loader_config = config[mode]["loader"]

    batch_size = loader_config["batch_size_per_card"]

    drop_last = loader_config["drop_last"]

    shuffle = loader_config["shuffle"]

    num_workers = loader_config["num_workers"]

    

    # Get collate_fn if specified

    if "collate_fn" in loader_config:

        collate_fn = getattr(collate_fn, loader_config["collate_fn"])()

    else:

        collate_fn = None



    if mode == "Train":

        # Check if using distributed training

        if torch.distributed.is_initialized():

            # Distributed training - multiple GPUs

            sampler = DistributedSampler(

                dataset,

                num_replicas=torch.distributed.get_world_size(),

                rank=torch.distributed.get_rank(),

                shuffle=shuffle

            )

            

            dataloader = DataLoader(

                dataset=dataset,

                batch_size=batch_size,

                sampler=sampler,

                num_workers=num_workers,

                drop_last=drop_last,

                pin_memory=True,

                collate_fn=collate_fn

            )

            

        else:

            # Single GPU or CPU training

            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

            

            dataloader = DataLoader(

                dataset=dataset,

                batch_size=batch_size,

                sampler=sampler,

                num_workers=num_workers,

                drop_last=drop_last,

                pin_memory=True,

                collate_fn=collate_fn

            )

            

    else:

        # Evaluation/Inference mode

        dataloader = DataLoader(

            dataset=dataset,

            batch_size=batch_size,

            shuffle=False,  # No shuffle for eval/test

            num_workers=num_workers,

            drop_last=drop_last,

            pin_memory=True,

            collate_fn=collate_fn

        )



    return dataloader




