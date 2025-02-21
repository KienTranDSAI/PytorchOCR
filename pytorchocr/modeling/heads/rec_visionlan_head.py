import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float32)
        sinusoid_table = sinusoid_table.unsqueeze(0)
        return sinusoid_table

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        self.pos_table = self.pos_table.to(device)
        return x + self.pos_table[:, : x.shape[1]]
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        k = k.transpose(1, 2)
        attn = torch.bmm(q, k)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1)
                mask = mask.unsqueeze(1)

            repeat_times = [
                attn.shape[1] // mask.shape[1],
                attn.shape[2] // mask.shape[2],
            ]
            
            mask = mask.repeat(1, repeat_times[0], repeat_times[1], 1)
            
            attn[mask == 0] = -1e9

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).reshape(-1, len_q, n_head, d_k)     # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).reshape(-1, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(-1, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).reshape(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).reshape(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).reshape(-1, len_v, d_v) # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1, 1) if mask is not None else None # (n*b) x .. x ..
        output = self.attention(q, k, v, mask=mask)
        output = output.reshape(n_head, -1, len_q, d_v)
        output = output.permute(1, 2, 0, 3).reshape(-1, len_q, n_head * d_v) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output
class Transformer_Encoder(nn.Module):
    def __init__(
            self,
        n_layers=2,
        n_head=8,
        d_word_vec=512,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_position=256,):
        super(Transformer_Encoder,self).__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, src_mask, return_attns=False):
        print(f"1.enc_output at transformer: {enc_output.shape}")
        enc_output = self.dropout(self.position_enc(enc_output))   # position embeding
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        print(f"enc_output at transformer: {enc_output.shape}")
        return enc_output
class PP_layer(nn.Module):
    def __init__(self,  n_dim=512, N_max_character=25, n_position=256):

        super(PP_layer, self).__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        print(f"n_dim: {n_dim}, n_position: {n_position}, n_max_char: {N_max_character}")

    def forward(self, enc_output):
        reading_order = torch.arange(self.character_len, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), self.character_len)    # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)      # b,25,512
        print(f"reading_order: {reading_order.shape}")
        print(f"enc_output: {enc_output.shape}")
        # calculate attention
        t = self.w0(reading_order.permute(0,2,1))
        print(f"self.wv: {self.wv}, self.w0: {self.w0}")
        print(f"t.permute(0,2,1): {t.permute(0,2,1).shape}, self.wv(enc_output): {self.wv(enc_output).shape}")
        
        t = self.active(t.permute(0,2,1) + self.wv(enc_output))     
        t = self.we(t)  # b,256,25
        t = self.softmax(t.permute(0,2,1))  # b,25,256
        g_output = torch.bmm(t, enc_output)  # b,25,512
        return g_output
class Prediction(nn.Module):
    def __init__(self, n_dim=512, n_class=37, N_max_character=25, n_position=256,):
        super(Prediction, self).__init__()
        self.pp = PP_layer(N_max_character=N_max_character, n_position=n_position,n_dim=n_dim)
        self.pp_share = PP_layer(n_dim=n_dim, N_max_character=N_max_character, n_position=n_position)
        self.w_vrm = nn.Linear(n_dim, n_class)    # output layer
        self.w_share = nn.Linear(n_dim, n_class)    # output layer
        self.nclass = n_class

    def forward(self, cnn_feature,  f_res, f_sub, train_mode=False, use_mlm = True):
        print(f"cnn_feature: {cnn_feature.shape}")
        if train_mode:
            if not use_mlm:
                g_output, attn = self.pp(cnn_feature)  # b,25,512
                g_output = self.w_vrm(g_output)
                f_res = 0
                f_sub = 0
                return g_output, f_res, f_sub
            g_output, attn = self.pp(cnn_feature)  # b,25,512
            f_res, _ = self.pp_share(f_res)
            f_sub, _ = self.pp_share(f_sub)
            g_output = self.w_vrm(g_output)
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            return g_output, f_res, f_sub
        else:
            g_output = self.pp(cnn_feature)  # b,25,512
            g_output = self.w_vrm(g_output)
            return g_output

class MLM(nn.Module):
    "Architecture of MLM"

    def __init__(self, n_dim=512, n_position=256, max_text_length=25):
        super(MLM, self).__init__()
        # Transformer Encoders for Sequence Modeling
        self.MLM_SequenceModeling_mask = Transformer_Encoder(
            n_layers=2, n_position=n_position
        )
        self.MLM_SequenceModeling_WCL = Transformer_Encoder(
            n_layers=1, n_position=n_position
        )
        self.pos_embedding = nn.Embedding(max_text_length, n_dim)
        self.w0_linear = nn.Linear(1, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.active = nn.Tanh()
        self.we = nn.Linear(n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, label_pos):
        # Transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(x, src_mask=None)
        
        # Position embedding layer
        label_pos = torch.tensor(label_pos, dtype=torch.long, device=x.device)
        pos_emb = self.pos_embedding(label_pos)
        pos_emb = self.w0_linear(pos_emb.unsqueeze(2))  # [B, S, 1]
        pos_emb = pos_emb.transpose(1, 2)  # [B, 1, S]
        
        # Fusion of position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        att_map_sub = self.we(att_map_sub)  # [B, 1, S]
        att_map_sub = att_map_sub.transpose(1, 2)  # [B, S, 1]
        att_map_sub = self.sigmoid(att_map_sub)  # [B, S, 1]
        
        # WCL (Word Classification Layer)
        att_map_sub = att_map_sub.transpose(1, 2)  # [B, 1, S]
        f_res = x * (1 - att_map_sub)  # Remaining string (with occluded chars)
        f_sub = x * att_map_sub  # Occluded characters
        
        # Transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)
        
        return f_res, f_sub, att_map_sub

def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    x = x.permute(0, 2, 1)
    x = x.reshape(-1, c, 32, 8)
    x = x.permute(0, 1, 3, 2)  # [16, 512, 8, 32]
    return x
class MLM_VRM(nn.Module):
    """
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling (transformer unit) + prediction layer (pp layer).
    x: input image
    label_pos: character index
    training_step: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    """

    def __init__(self, n_layers=3, n_position=256, n_dim=512, max_text_length=25, nclass=37):
        super(MLM_VRM, self).__init__()
        self.MLM = MLM(n_dim=n_dim, n_position=n_position, max_text_length=max_text_length)
        self.SequenceModeling = Transformer_Encoder(n_layers=n_layers, n_position=n_position)
        self.Prediction = Prediction(
            n_dim=n_dim,
            n_position=n_position,
            N_max_character=max_text_length + 1,  # N_max_character = 1 eos + 25 characters
            n_class=nclass,
        )
        self.nclass = nclass
        self.max_text_length = max_text_length

    def forward(self, x, label_pos, training_step, train_mode=False):
        b, c, h, w = x.shape
        nT = self.max_text_length
        x = x.permute(0, 1, 3, 2)  # [B, C, H, W] -> [B, C, W, H]
        x = x.reshape(-1, c, h * w)
        x = x.transpose(1, 2)  # [B, W*H, C]

        if train_mode:
            if training_step == "LF_1":
                f_res = 0
                f_sub = 0
                x = self.SequenceModeling(x,src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(x, f_res, f_sub, train_mode=True, use_mlm=False)
                return text_pre, text_pre, text_pre, text_pre
            elif training_step == "LF_2":
                # MLM
                f_res, f_sub, mask_c = self.MLM(x, label_pos)
                x = self.SequenceModeling(x, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(x, f_res, f_sub, train_mode=True)
                mask_c_show = trans_1d_2d(mask_c)
                return text_pre, test_rem, text_mas, mask_c_show
            elif training_step == "LA":
                # MLM
                f_res, f_sub, mask_c = self.MLM(x, label_pos)
                character_mask = torch.zeros_like(mask_c)

                ratio = b // 2
                if ratio >= 1:
                    with torch.no_grad():
                        character_mask[0:ratio, :, :] = mask_c[0:ratio, :, :]
                else:
                    character_mask = mask_c
                x = x * (1 - character_mask)
                # VRM
                x = self.SequenceModeling(x, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(x, f_res, f_sub, train_mode=True)
                mask_c_show = trans_1d_2d(mask_c)
                return text_pre, test_rem, text_mas, mask_c_show
            else:
                raise NotImplementedError
        else:  # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            print(f"x: {x.shape}")
            contextual_feature = self.SequenceModeling(x, src_mask=None)
            print(f"contextual_feature: {contextual_feature.shape}")
            text_pre = self.Prediction(contextual_feature, f_res, f_sub, train_mode=False, use_mlm=False)
            text_pre = text_pre.transpose(1, 0)  # (26, b, 37) -> [S, B, n_class]
            return text_pre, x
class VLHead(nn.Module):
    """
    Architecture of VisionLAN
    """

    def __init__(
        self,
        in_channels,
        out_channels=36,
        n_layers=3,
        n_position=256,
        n_dim=512,
        max_text_length=25,
        training_step="LA",
    ):
        super(VLHead, self).__init__()
        self.MLM_VRM = MLM_VRM(
            n_layers=n_layers,
            n_position=n_position,
            n_dim=n_dim,
            max_text_length=max_text_length,
            nclass=out_channels,
        )
        self.training_step = training_step

    def forward(self, feat, targets=None):
        if self.training:
            label_pos = targets[-2]  # Assuming `targets[-2]` is the label position
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(
                feat, label_pos, self.training_step, train_mode=True
            )
            return text_pre, test_rem, text_mas, mask_map
        else:
            print(f"feat: {feat.shape}")
            print(f"target: {targets}")
            text_pre, x = self.MLM_VRM(
                feat, targets, self.training_step, train_mode=False
            )
            return text_pre, x
