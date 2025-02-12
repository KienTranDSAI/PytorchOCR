import os
import torch
import pickle
import json
def fast_load_model(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print(f"Load checkpoint successfully!")
    
def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == 17 and os.path.isdir(path):  # 17 corresponds to EEXIST error
                print(f"Be happy if some process has already created {path}")
            else:
                raise OSError(f"Failed to mkdir {path}")

def load_model(config, model, optimizer=None, model_type="det"):
    """
    Load model from checkpoint or pretrained_model
    """
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
    is_float16 = False


    if checkpoints:
        assert os.path.exists(checkpoints + '.pth'), f"The {checkpoints + '.pth'} does not exist!"

        # Load params from trained model
        params = torch.load(checkpoints + '.pth')
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                print(f"{key} not in loaded params!")
                continue
            pre_value = params[key]
            if pre_value.dtype == torch.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.to(value.dtype)
            if value.shape == pre_value.shape:
                new_state_dict[key] = pre_value
            else:
                print(f"Shape mismatch for {key}: {value.shape} vs {pre_value.shape}")
        model.load_state_dict(new_state_dict)

        if optimizer is not None:
            optim_path = checkpoints + ".optim.pth"
            if os.path.exists(optim_path):
                optimizer.load_state_dict(torch.load(optim_path))
            else:
                print(f"{optim_path} not found, optimizer params not loaded")

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            best_model_dict["acc"] = 0.0
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        print(f"Resuming from {checkpoints}")
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model)
    else:
        print("Training from scratch")
    best_model_dict["is_float16"] = is_float16
    return best_model_dict
def maybe_download_params(path):

    """Check if parameter file exists, return the path directly if it does"""

    if os.path.exists(path + '.pth'):

        return path

    else:

        # For now, just return the path as-is since we're not implementing download logic

        return path
def load_pretrained_params(model, path):
    path = maybe_download_params(path)
    assert os.path.exists(path + ".pth"), f"The {path}.pth does not exist!"

    params = torch.load(path + ".pth")
    state_dict = model.state_dict()
    new_state_dict = {}
    is_float16 = False

    for k1 in params.keys():
        if k1 not in state_dict.keys():
            print(f"The pretrained params {k1} not in model")
        else:
            if params[k1].dtype == torch.float16:
                is_float16 = True
            if params[k1].dtype != state_dict[k1].dtype:
                params[k1] = params[k1].to(state_dict[k1].dtype)
            if state_dict[k1].shape == params[k1].shape:
                new_state_dict[k1] = params[k1]
            else:
                print(f"Shape mismatch for {k1}: {state_dict[k1].shape} vs {params[k1].shape}")

    model.load_state_dict(new_state_dict)
    if is_float16:
        print("The parameter type is float16, which is converted to float32 when loading")
    print(f"Pretrained model loaded successfully from {path}")
    return is_float16

def save_model(model, optimizer, model_path, config, is_best=False, prefix="model", **kwargs):
    """
    Save model to the target path
    """
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)

    if prefix == "best_accuracy":
        best_model_path = os.path.join(model_path, "best_model")
        _mkdir_if_not_exist(best_model_path)

    torch.save(optimizer.state_dict(), model_prefix + ".optim.pth")
    if prefix == "best_accuracy":
        torch.save(optimizer.state_dict(), os.path.join(best_model_path, "model.optim.pth"))

    torch.save(model.state_dict(), model_prefix + ".pth")
    if prefix == "best_accuracy":
        torch.save(model.state_dict(), os.path.join(best_model_path, "model.pth"))

    save_model_info = kwargs.pop("save_model_info", False)
    if save_model_info:
        with open(os.path.join(model_path, f"{prefix}.info.json"), "w") as f:
            json.dump(kwargs, f)
        print(f"Model info saved in {model_path}")

    # Save metric and config
    with open(model_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        print(f"Best model saved to {model_prefix}")
    else:
        print(f"Model saved to {model_prefix}")


def main():
    # Example main function to use these methods
    config = {
        "Global": {
            "checkpoints": "path/to/your/checkpoint",  # Replace with actual path
            "pretrained_model": "path/to/pretrained/model",  # Replace with actual path
        },
        "Architecture": {
            "algorithm": "Distillation",
            "Backbone": {"checkpoints": "path/to/backbone/"},
        },
    }
    model = torch.nn.Module()  # Replace with actual model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    load_model(config, model, optimizer)
    save_model(model, optimizer, "path/to/save", config)

if __name__ == "__main__":
    main()
