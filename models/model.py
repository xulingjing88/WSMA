import torch
import torch.nn as nn
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
import numpy as np
from torch.nn import functional as F
from AIM.modules.burger import HamburgerV1
from collections import OrderedDict
from pkg_resources import packaging
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Any, Union, List

_tokenizer = _Tokenizer()

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PromptLearner(nn.Module):
    def __init__(self, classnames, ln_final, token_embedding):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = ln_final.weight.dtype
        ctx_dim = ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x_1 = x + self.positional_embedding[:, None, :].to(x.dtype)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x[0], x[1:], att[:, 1:, 1:]
    
class Cross_Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj_q = nn.Linear(in_dim, out_dim)
        self.proj_k = nn.Linear(in_dim, out_dim)
        self.proj_v = nn.Linear(in_dim, out_dim)
        self.scale = self.out_dim ** (-0.5)

        self.norm = nn.LayerNorm(self.in_dim)
    def forward(self, ego, exo):

        B, hw, C = ego.size()
        query = self.proj_q(ego)                                    
        key = self.proj_k(exo)                                       
        value = self.proj_v(exo)

        att = torch.bmm(query, key.transpose(1, 2))*self.scale               
        att = att.softmax(dim=-1)
        out = torch.bmm(att, value)

        out = self.norm(out+ego)                               

        return out.transpose(1, 2).view(B, C, 14, 14)
    
class Model(nn.Module):
    def __init__(self, args, embed_dim:int, context_length: int, vocab_size: int, 
                 transformer_width: int, transformer_heads: int, 
                 transformer_layers: int, num_classes=36, pretrained=True, n=3, D=512):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.pretrained = pretrained
        self.n = n
        self.D = D
        self.context_length = context_length
        if args.divide == "Seen":
            self.classnames = ['beat', "boxing", "brush_with", "carry", "catch",
                         "cut", "cut_with", "drag", 'drink_with', "eat",
                         "hit", "hold", "jump", "kick", "lie_on", "lift",
                         "look_out", "open", "pack", "peel", "pick_up",
                         "pour", "push", "ride", "sip", "sit_on", "stick",
                         "stir", "swing", "take_photo", "talk_on", "text_on",
                         "throw", "type_on", "wash", "write"]
        elif args.divide=="Unseen":
            self.classnames = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
        else: # HICO-IIF
            self.classnames = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']
        
        # dino-vit
        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16
        self.Hamburger = HamburgerV1(in_c=self.vit_feat_dim, n=self.n, D=self.D)
        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        self.aff_proj = Mlp(in_features=int(self.vit_feat_dim*2), hidden_features=int(self.vit_feat_dim*2), out_features=self.vit_feat_dim,
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        
        # clip
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.attnpool = AttentionPool2d(14, self.vit_feat_dim, 64, embed_dim)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.prompt_learner = PromptLearner(self.classnames, self.ln_final, self.token_embedding)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # fc
        self.fc = nn.Linear(self.vit_feat_dim, self.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def encode_text(self, per, text):
        x = per + self.token_embedding(text.cuda()).float()  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).float()

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward(self, exocentric, egocentric_image, label, text):
        target = label.long().squeeze()
        b, n, c, h, w = exocentric.size()
        exocentrin_input = exocentric.view(b * n, c, h, w)
        
        # dino_vit
        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_all_key(egocentric_image)
            _, exo_key, exo_attn = self.vit_model.get_all_key(exocentrin_input)
            ego_desc = ego_key[len(ego_key)-2].permute(0, 2, 3, 1).flatten(-2, -1).detach()
            exo_desc = exo_key[len(ego_key)-2].permute(0, 2, 3, 1).flatten(-2, -1).detach()
            for i in range(len(ego_key)-1, len(ego_key)):
                ego_desc = torch.cat((ego_desc, ego_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)
                exo_desc = torch.cat((exo_desc, exo_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)

        ego_proj = self.aff_proj(ego_desc[:, 1:])
        exo_proj = self.aff_proj(exo_desc[:, 1:])
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)
        
        exo_proj = self.Hamburger(exo_proj)
        
        # text branch
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.encode_text(prompts, tokenized_prompts)
        
        e_b, e_c, e_h, e_w = ego_proj.shape
        pre_ego = ego_proj
        image_features, ego_proj, mu_att = self.attnpool(ego_proj)
        
        image_features = F.normalize(image_features, dim=1, p=2)
        
        text_features = F.normalize(text_features, dim=1, p=2)
        
        
        # Pixel-Text Fusion
        logit_scale = self.logit_scale.exp()
        self.logits_per_image = logit_scale * image_features @ text_features.t()
        self.logits_per_text = self.logits_per_image.t()
        
        text_f = torch.ones((e_b, 1024)).cuda()
        for i in range(e_b):
            text_f[i] = text_features[label[i]]
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = logit_scale *att_egoproj.permute(1, 0, 2)@text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)
        ego_proj = attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w)*pre_ego + pre_ego
        
        exocentric_branch =self.aff_exo_proj(exo_proj)
        egocentric_branch =self.aff_ego_proj(ego_proj)
        
        # cls
        exo_pool = self.avgpool(exocentric_branch)
        exo_pool = exo_pool.view(exo_pool.size(0), -1)
        self.exo_score = self.fc(exo_pool)
        
        batch, channel, h, w = exocentric_branch.shape
        exocentric_branch = exocentric_branch.view(batch//3, 3, channel, h, w).mean(1)
        batch = batch//3
        
        exo_weight = self.fc.weight[target]
        exo_weight = exo_weight.view(batch, channel, 1, 1).expand_as(exocentric_branch)
        self.exo_feature = (exo_weight * exocentric_branch)
        
        self.exo_features = torch.ones(batch, self.num_classes, e_h, e_w).cuda()
        label_sum = torch.ones_like(label).cuda()
        for m in range(0,self.num_classes):
            weight = self.fc.weight[label_sum.long()*m]
            weight = weight.view(batch, channel, 1, 1).expand_as(exocentric_branch)
            self.exo_features[:,m] = (weight * exocentric_branch).mean(1)
        
        ego_pool = self.avgpool(egocentric_branch)
        ego_pool = ego_pool.view(ego_pool.size(0), -1)
        self.ego_score = self.fc(ego_pool)

        ego_weight = self.fc.weight[target]
        ego_weight = ego_weight.view(batch, channel, 1, 1).expand_as(egocentric_branch)
        self.ego_feature = (ego_weight * egocentric_branch)
        
        self.ego_features = torch.ones(batch, self.num_classes, e_h, e_w).cuda()
        label_sum = torch.ones_like(label).cuda()
        for m in range(0,self.num_classes):
            gweight = self.fc.weight[label_sum.long()*m]
            gweight = gweight.view(batch, channel, 1, 1).expand_as(egocentric_branch)
            self.ego_features[:,m] = (gweight * egocentric_branch).mean(1)
            
        # l_rela
        self.exo_att = self.exo_features.view(batch, self.num_classes, -1).transpose(1, 2)
        self.exo_att = torch.matmul(self.exo_features.view(batch, self.num_classes, -1), self.exo_att)
        self.ego_att = self.ego_features.view(batch, self.num_classes, -1).transpose(1, 2)
        self.ego_att = torch.matmul(self.ego_features.view(batch, self.num_classes, -1), self.ego_att)
            
        return self.exo_score, self.ego_score, self.logits_per_text, self.logits_per_image
    
    @torch.no_grad()
    def get(self, egocentric_image, label, text):
        
        # dino_vit
        _, ego_key, ego_attn = self.vit_model.get_all_key(egocentric_image)
        ego_desc = ego_key[len(ego_key)-2].permute(0, 2, 3, 1).flatten(-2, -1).detach()
        for i in range(len(ego_key)-1, len(ego_key)):
            ego_desc = torch.cat((ego_desc, ego_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)
        ego_proj = self.aff_proj(ego_desc[:, 1:])
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        
        # text branch
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.encode_text(prompts, tokenized_prompts)
        
        e_b, e_c, e_h, e_w = ego_proj.shape
        pre_ego = ego_proj
        image_features, ego_proj, mu_att = self.attnpool(ego_proj)
        
        image_features = F.normalize(image_features, dim=1, p=2)
        text_features = F.normalize(text_features, dim=1, p=2)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        text_f = torch.ones((e_b, 1024)).cuda()
        for i in range(e_b):
            text_f[i] = text_features[label[i]]
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = att_egoproj.permute(1, 0, 2)@text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)
        ego_proj = attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w)*pre_ego + pre_ego
        
        mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
        mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
        for _ in range(2):
            mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
            mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
        mu_att = (mu_att + mu_att.permute(0, 2, 1)) / 2
        mu_att = torch.matmul(mu_att, mu_att)
        
        egocentric_branch =self.aff_ego_proj(ego_proj)
        
        ego_pool = self.avgpool(egocentric_branch)
        ego_pool = ego_pool.view(ego_pool.size(0), -1)
        self.ego_score = self.fc(ego_pool)
        
        target = label.long().squeeze()
        batch, channel,_,_ = egocentric_branch.shape
        
        cam_weight = self.fc.weight[target]
        cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(egocentric_branch)
        cam = (cam_weight * egocentric_branch).mean(1)
        
        cam1 = mu_att@(cam.view(batch, -1, 1))
        cam1 = cam1.view(batch, e_h, e_w)
        
        return cam, cam1
    
    def get_loss(self, gt_label, separate=False):
        # loss L_cls
        b, h = self.exo_score.shape
        self.exo_score = self.exo_score.view(b//3, -1, h)
        loss_cls = (self.criterion(self.ego_score, gt_label)+ (self.criterion(self.exo_score[:, 0], gt_label)
                                                               + self.criterion(self.exo_score[:, 1], gt_label)
                                                               + self.criterion(self.exo_score[:, 2], gt_label))/3)
        # loss L_d
        exo_branch, ego_branch = self.exo_feature,self.ego_feature
        exo_pool = F.adaptive_avg_pool2d(exo_branch, 1).view(exo_branch.size(0), -1)
        exo_pool = F.normalize(exo_pool, 2, dim=1)
        ego_pool = F.adaptive_avg_pool2d(ego_branch, 1).view(ego_branch.size(0), -1)
        ego_pool = F.normalize(ego_pool, 2, dim=1)
        loss_dist =0.5 * (((exo_pool - ego_pool) ** 2).sum(1)).mean(0)
        
        # loss L_lrela
        exo_att, ego_att = self.exo_att,self.ego_att
        attb, c, _ = exo_att.size()
        exo_att = F.normalize(exo_att, 2, dim=2).view(attb, -1)
        ego_att = F.normalize(ego_att, 2, dim=2).view(attb, -1)
        loss_att = 0.5*(1 - F.cosine_similarity(exo_att, ego_att, dim=1).mean(0))

        return loss_cls, loss_dist, loss_att
    
    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens

        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

def MODEL(args, num_classes=36,
          pretrained=True, n=3, D=512):
    dict = 'RN50.pt' # clip's pre-trained model
    state_dict =  torch.jit.load(dict)
    state_dict = state_dict.state_dict()
    
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = Model(args, embed_dim=embed_dim, context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
                  transformer_heads=transformer_heads,transformer_layers=transformer_layers, num_classes=num_classes, pretrained=pretrained, n=n, D=D)

    model_dict = model.state_dict()
    par = []
    pretrained_dict = {}
    for para in model.named_parameters():
        k = para[0]
        if k in state_dict:
            par.append(para[0])
    for k, v in state_dict.items(): 
        if k in model_dict:
            pretrained_dict[k] = v
        
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model, par
