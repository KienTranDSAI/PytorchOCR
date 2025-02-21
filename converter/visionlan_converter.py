# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class PPOCRv3RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        para_state_dict = self.del_invalid_state_dict(para_state_dict)
        total_params = sum(p.size if isinstance(p, np.ndarray) else p.numel() for p in para_state_dict.values())
        print(f'Total number of parameters in para_state_dict: {total_params}')
        out_channels = list(para_state_dict.values())[-1].shape[0]
        print('out_channels: ', out_channels)
        print(type(kwargs), kwargs)
        kwargs['out_channels'] = out_channels
        super(PPOCRv3RecConverter, self).__init__(config, **kwargs)
        # self.load_paddle_weights(paddle_pretrained_model_path)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        otal_params = sum(p.numel() for p in self.net.parameters())
        print(f'Total number of parameters in self.net: {total_params}')
        self.net.eval()


    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        for i, (k,v) in enumerate(para_state_dict.items()):
            if k.startswith('Teacher.'):
                continue

            elif k.startswith('Student.head.sar_head.'):
                continue

            else:
                new_state_dict[k] = v
        return new_state_dict


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        [print('paddle: {} ---- {}'.format(k, v.shape)) for k, v in para_state_dict.items()]
        # [print('pytorch: {} ---- {}'.format(k, v.shape)) for k, v in self.net.state_dict().items()]

        for k,v in self.net.state_dict().items():
            if k.endswith('num_batches_tracked'):
                continue

            ppname = k
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')
            ppname = ppname.replace('neck.encoder.', 'head.ctc_encoder.encoder.')
            ppname = ppname.replace('head.fc.', 'head.ctc_head.fc.')

            try:
                try:
                    if ppname.endswith('fc1.weight') or ppname.endswith('fc2.weight') \
                            or ppname.endswith('fc.weight') or ppname.endswith('qkv.weight') \
                            or ppname.endswith('proj.weight') or ppname.endswith("MLM_VRM.MLM.w0_linear.weight") \
                                or ppname.endswith("MLM_VRM.MLM.we.weight") \
                                    or ppname.endswith("MLM_VRM.Prediction.pp.w0.weight") \
                                        or ppname.endswith("MLM_VRM.Prediction.pp.we.weight") or ("weight" in ppname and "MLM_VRM") \
                                            and ("backbone" not in ppname) \
                                                and ("MLM_SequenceModeling_mask" not in ppname) \
                                                    and ("MLM_SequenceModeling_WCL" not in ppname): # or ("weight" in ppname and "MLM_VRM")
                        # print("Transpose!")
                        self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                    else:
                        # print("No transpose")

                        self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
                except:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

        print('model is loaded.')


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'rec',
           'algorithm':'VisionLAN',
           'Transform':None,
           'Backbone':{'name':'ResNet45',
                       'strides': [2, 2, 2, 1, 1]},
           'Head':{'name':'VLHead',
            "n_layers": 3,
            "n_position": 256,
            "n_dim": 512,
            "max_text_length": 40,
            "training_step": 'LA'}
           }
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPOCRv3RecConverter(cfg, paddle_pretrained_model_path)
    
    np.random.seed(666)
    inputs = np.random.randn(1,3,48,320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    
    save_basename = os.path.basename(os.path.abspath(args.src_model_path))

    save_name = 'visionlan_ver1.pth'
    converter.save_pytorch_weights(save_name)
    print('done.')
    print(converter.net)



####Model path: /raid/kientdt/shared_drive_cv/ocr/kientdt/PaddleOCR/output/rec/r45_visionlan