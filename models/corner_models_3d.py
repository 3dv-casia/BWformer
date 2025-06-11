import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableAttnDecoderLayer, DeformableTransformerDecoderLayer
from models.ops.modules import MSDeformAttn
from models.resnet import convrelu
from models.matcher import build_matcher
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from einops.layers.torch import Rearrange
from utils.misc import NestedTensor

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class HeatCorner3d(nn.Module):
    """
        The corner model of HEAT is the edge model till the edge-filtering part. So only per-candidate prediction w/o
    relational modeling.
    """
    def __init__(self, input_dim, hidden_dim, num_feature_levels, backbone_strides, backbone_num_channels, ):
        super(HeatCorner3d, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        self.center_embed = MLP(hidden_dim, hidden_dim, 1, 3)

        nn.init.constant_(self.center_embed.layers[-1].bias.data[2:], -2.0)
        num_classes = 1
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.pos3d_embed = nn.Linear(3, 2)

        self.pos_trans_norm = nn.LayerNorm(hidden_dim)
        self.pos_trans = nn.Linear(192, hidden_dim)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.num_queries = 300
        
        self.query_embed = nn.Embedding(self.num_queries, 1)
        self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)


        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.patch_size = 4
        patch_dim = (self.patch_size ** 2) * input_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        self.pixel_pe_fc = nn.Linear(input_dim, hidden_dim)
        self.transformer = Corner3dTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
                                             dim_feedforward=1024, dropout=0.1)

        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)

    @staticmethod
    def get_ms_feat(xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    @staticmethod
    def get_decoder_reference_points(height, width, device):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref = torch.stack((ref_x, ref_y), -1)
        return ref

    def forward(self, corners2d,image_feats, feat_mask,  all_image_feats):
        # process image features
        features = self.get_ms_feat(image_feats, feat_mask)

        srcs = []
        masks = []
        all_pos = []

        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)


        query_embeds3d = self.query_embed.weight
        query_embeds3d = query_embeds3d.unsqueeze(0).expand(corners2d.shape[0], -1, -1)
        
        
        tgt_embeds = self.tgt_embed.weight
        query_embeds3d_sig = query_embeds3d.sigmoid()
        corners2d = (corners2d/255)
        div_term = torch.exp( torch.arange(0, 192 // 3, 2, dtype=torch.float, device = query_embeds3d.device) *  (-math.log(10000.0) / (192 // 3)))
        div_term = div_term.view( 1, -1) 
        
        x_position, y_position, z_position = corners2d[:,:, 0:1] , corners2d[:,:,1:2], query_embeds3d_sig[:,:,0:1]
        
        sinx = torch.sin(x_position * div_term) # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        position_code = torch.cat( [ sinx, cosx, siny, cosy, sinz, cosz] , dim=-1 )
        position_code = self.pos_trans_norm(self.pos_trans(position_code))

 
        query_embeds = corners2d


        corner_hs, corner_reference_points = self.transformer(srcs, masks, all_pos, tgt_embeds, query_embeds, all_image_feats,query_pos_embed=position_code)
        
        corner_logits = self.class_embed(corner_hs)
        corner_coord = self.center_embed(corner_hs) 

        corner_coord += query_embeds3d
        corner_coord = corner_coord.sigmoid()
        corner_coord = torch.cat((corners2d, corner_coord), dim=-1)

        
        return corner_logits, corner_coord
        

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).bool().to(x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



class Corner3dTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 ):
        super(Corner3dTransformer, self).__init__()

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

      
        decoder_attn_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 6, False, with_sa=True)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = convrelu(256 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, d_model, 3, 1)
        self.output_fc_1 = nn.Linear(d_model, 1)
        self.output_fc_2 = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, reference_points, all_image_feats, query_pos_embed):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        bs, _, c = memory.shape
        
        tgt = query_embed
        reference_points2 = reference_points
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points2
        get_image_feat = True
        keypad = None
        hs_pixels_s1, reference_points = self.per_edge_decoder(tgt, reference_points, memory,
                                                 spatial_shapes, level_start_index, valid_ratios, query_pos_embed,
                                                 mask_flatten, keypad,  get_image_feat)
       
        return hs_pixels_s1, reference_points

    def generate_corner_preds(self, outputs, conv_outputs):
        B, L, C = outputs.shape
        side = int(np.sqrt(L))
        outputs = outputs.view(B, side, side, C)
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = torch.cat([outputs, conv_outputs['layer1']], dim=1)
        x = self.conv_up1(outputs)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['layer0']], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, conv_outputs['x_original']], dim=1)
        x = self.conv_original_size2(x)

        logits = x.permute(0, 2, 3, 1)
        preds = self.output_fc_1(logits)
        preds = preds.squeeze(-1).sigmoid()
        return logits, preds
    
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


