import copy
import torch
from torch.optim import lr_scheduler

__all__ = ["build_optimizer"]

def build_lr_scheduler(optimizer, lr_config, epochs, step_each_epoch):
    """Build PyTorch learning rate scheduler"""
    lr_config = copy.deepcopy(lr_config)
    lr_name = lr_config.pop("name", "Const").lower()
    
    total_steps = epochs * step_each_epoch
    warmup_steps = round(lr_config.pop("warmup_epoch", 0) * step_each_epoch)
    
    # Get base learning rate
    base_lr = lr_config.pop("learning_rate", 0.001)
    
    # Configure scheduler based on name
    if lr_name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr_config.pop("end_lr", 0)
        )
    elif lr_name == "step":
        step_size = int(lr_config.pop("step_size", 1) * step_each_epoch)
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=lr_config.pop("gamma", 0.1)
        )
    elif lr_name == "multistep":
        milestones = [int(x * step_each_epoch) for x in lr_config.pop("milestones", [2, 4])]
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=lr_config.pop("gamma", 0.1)
        )
    else:  # default to constant LR
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda _: 1)

    # Add warmup if specified
    if warmup_steps > 0:
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=lr_config.pop("start_lr", base_lr/3) / base_lr,
                    end_factor=1.0,
                    total_iters=warmup_steps
                ),
                scheduler
            ],
            milestones=[warmup_steps]
        )

    return scheduler

def build_optimizer(config, epochs, step_each_epoch, model):
    """Build PyTorch optimizer and learning rate scheduler"""
    config = copy.deepcopy(config)
    
    # Get learning rate config
    lr_config = config.pop('lr')
    base_lr = lr_config.get("learning_rate", 0.001)
    
    # Configure optimizer parameters
    weight_decay = config.pop('weight_decay', 0)
    optim_name = config.pop('name').lower()
    
    # Handle gradient clipping
    clip_norm = config.pop('clip_norm', None) or config.pop('clip_norm_global', None)
    
    # Prepare optimizer parameters
    no_decay_keys = config.pop('no_weight_decay_name', '').split()
    params = []
    
    if no_decay_keys:
        # Split parameters into decay/no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in no_decay_keys):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    else:
        params = model.parameters()

    # Create optimizer
    if optim_name == "momentum":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=config.pop('momentum', 0.9),
            weight_decay=weight_decay
        )
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(config.pop('beta1', 0.9), config.pop('beta2', 0.999)),
            weight_decay=weight_decay
        )
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            betas=(config.pop('beta1', 0.9), config.pop('beta2', 0.999)),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    # Create learning rate scheduler
    scheduler = build_lr_scheduler(optimizer, lr_config, epochs, step_each_epoch)

    # Add gradient clipping if specified
    if clip_norm:
        def clip_gradients():
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.clip_gradients = clip_gradients

    return optimizer, scheduler
