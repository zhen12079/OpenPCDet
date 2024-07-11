#coding=utf-8
#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from os.path import join as pjoin
from scipy import ndimage
from einops.layers.torch import Rearrange

import logging
import math
import copy
# from baseline.models.registry import BACKBONE

# logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, hidden_size=512,  # 输入token的dim
                       num_heads=16,
                       attn_drop_ratio=0.,
                       proj_drop_ratio=0., 
                       vis=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)
        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attn_drop_ratio)
        self.proj_dropout = Dropout(proj_drop_ratio)

        self.softmax = Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)
    
    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attnetion_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attnetion_scores = attnetion_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attnetion_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights
    

class Mlp(nn.Module):
    def __init__(self, hidden_size=512,  # 输入token的dim
                       mlp_dim=2048,
                       drop_ratio=0.1,) -> None:
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(drop_ratio)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=144,
                       patch_size=8,
                       in_channels=64,
                       hidden_size=512,
                       dropout_rate=0.1,
                       cls_token=False):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        self.patch_embeddings = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )
        self.num_tokens = 1 if cls_token else 0
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+self.num_tokens, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        if self.cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        if self.cls_token:
            x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    

class Block(nn.Module):
    def __init__(self, hidden_size=512, 
                       mlp_dim=2048,
                       num_heads=16,
                       drop_ratio=0.1,
                       attn_drop_ratio=0.,
                       proj_drop_ratio=0.,
                       vis=None):
        super(Block, self).__init__()

        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size=hidden_size,
                       mlp_dim=mlp_dim,
                       drop_ratio=drop_ratio)
        self.attn = Attention(hidden_size=hidden_size,
                              num_heads=num_heads,
                              attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=proj_drop_ratio, 
                              vis=vis)
    
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
    
class Encoder(nn.Module):
    def __init__(self, hidden_size=512, 
                       mlp_dim=2048,
                       num_heads=16,
                       num_layers=6,
                       attn_drop_ratio=0.,
                       proj_drop_ratio=0.,
                       drop_ratio=0.1,
                       vis=None):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size=hidden_size, 
                          mlp_dim=mlp_dim,
                          num_heads=num_heads,
                          drop_ratio=drop_ratio,
                          attn_drop_ratio=attn_drop_ratio,
                          proj_drop_ratio=proj_drop_ratio,
                          vis=vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)

        encoded = self.encoder_norm(hidden_states)

        return encoded, attn_weights
    

class Transformer(nn.Module):
    def __init__(self, img_size=144,
                       patch_size=8,
                       in_channels=64,
                       hidden_size=512,
                       num_heads=16,
                       mlp_dim=2048,
                       num_layers=6,
                       dropout_rate=0.1,
                       attn_drop_ratio=0.,
                       proj_drop_ratio=0.,
                       vis=None
                 ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size,
                                    patch_size=patch_size,
                                    in_channels=in_channels,
                                    hidden_size=hidden_size,
                                    dropout_rate=dropout_rate)
        
        self.encoder = Encoder(hidden_size=hidden_size, 
                                mlp_dim=mlp_dim,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                drop_ratio=dropout_rate,
                                attn_drop_ratio=attn_drop_ratio,
                                proj_drop_ratio=proj_drop_ratio,
                                vis=vis)
        
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    
# @BACKBONE.register_module
class VisionTransformer(nn.Module):
    def __init__(self, img_size=144,
                       patch_size=8,
                       in_channels=64,
                       hidden_size=512,
                       num_layers=6,
                       num_heads=16,
                       mlp_ratio=4,
                       drop_ratio=0.1,
                       attn_drop_ratio=0.0,
                       drop_path_ratio=0.0,
                       zero_head=False,
                       vis=False,
                       cfg=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            num_classes (int): number of classes for classification head
            hidden_size (int): embedding dimension
            num_layers (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
        """
        super(VisionTransformer, self).__init__()
        self.zero_head = zero_head
        mlp_dim = int(hidden_size*mlp_ratio)
        self.transformer = Transformer(img_size=img_size,
                                        patch_size=patch_size,
                                        in_channels=in_channels,
                                        hidden_size=hidden_size,
                                        num_heads=num_heads,
                                        mlp_dim=mlp_dim,
                                        num_layers=num_layers,
                                        dropout_rate=drop_ratio,
                                        attn_drop_ratio=attn_drop_ratio,
                                        proj_drop_ratio=drop_path_ratio,
                                        vis=vis)
        
        temp_h = int(img_size/patch_size)
        self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = temp_h, p1 = patch_size, p2 = patch_size)
        
        out_in_channels = int(hidden_size/(patch_size**2))
        output_channels = 1024
        is_with_shared_mlp = True
        if is_with_shared_mlp:
            self.is_with_shared_mlp = True
            self.shared_mlp = nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1)
        else:
            self.is_with_shared_mlp = False

    def forward(self, x):
        x, attn_weights = self.transformer(x)


        x = self.rearrange(x)
        
        if self.is_with_shared_mlp:
            x = self.shared_mlp(x)

        return x, attn_weights
    
    def load_from(self, weights):
        '''for load pretrained_dir
        model.load_from(np.load(args.pretrained_dir))
        '''
        with torch.no_grad():
            # if self.zero_head:
            #     nn.init.zeros_(self.head.weight)
            #     nn.init.zeros_(self.head.bias)
            # else:
            #     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
            #     self.head.bias.copy_(np2th(weights["head/bias"]).t())
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                # logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # if self.transformer.embeddings.hybrid:
            #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
            #         for uname, unit in block.named_children():
            #             unit.load_from(weights, n_block=bname, n_unit=uname)


if __name__=="__main__":

    vit = VisionTransformer(img_size=144,
                            patch_size=8,
                            in_channels=64,
                            hidden_size=512,
                            num_layers=6,
                            drop_ratio=0.1,
                            zero_head=False,
                            vis=True,
                            cfg=None)
    
    img = torch.randn(1, 64, 144, 144).cuda()
    vit.to('cuda')
    preds = vit(img)
    print(preds[0].shape)
    # print(preds[1], len(preds[1]))
