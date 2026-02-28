import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from clip.model import CLIP
from model.utils import low_rank_approx  

from .peft_modules import *


class Text_Tuner(nn.Module):
    def __init__(self, cfg, clip_model: CLIP):
        super().__init__()

        self.cfg = cfg

        n_layers = len(clip_model.transformer.resblocks)
        emb_dim = clip_model.positional_embedding.shape[1]
        dtype = clip_model.text_projection.dtype

        blocks = clip_model.transformer.resblocks

        get_attn_in_weight = lambda i: blocks[i].attn.in_proj_weight
        get_attn_in_bias = lambda i: blocks[i].attn.in_proj_bias
        get_attn_out_weight = lambda i: blocks[i].attn.out_proj.weight
        get_attn_out_bias = lambda i: blocks[i].attn.out_proj.bias
        get_mlp_in_weight = lambda i: blocks[i].mlp[0].weight
        get_mlp_in_bias = lambda i: blocks[i].mlp[0].bias
        get_mlp_out_weight = lambda i: blocks[i].mlp[2].weight
        get_mlp_out_bias = lambda i: blocks[i].mlp[2].bias

        attn_in_dim = get_attn_in_bias(0).shape[0]
        attn_out_dim = get_attn_out_bias(0).shape[0]
        mlp_in_dim = get_mlp_in_bias(0).shape[0]
        mlp_out_dim = get_mlp_out_bias(0).shape[0]


        use_full_tuning = cfg.t_full_tuning
        partial = cfg.t_partial

        use_keeplora = cfg.t_keeplora
        lora_rank = cfg.t_adapter_dim

        if partial is None:
            _start, _end = 0, n_layers
        elif isinstance(partial, int):
            _start, _end = n_layers - partial, n_layers
        elif isinstance(partial, list):
            _start, _end = partial[0], partial[1]

        if use_full_tuning:
            block_tuned = blocks[_start: _end]
        else:
            block_tuned = None

        # Initialize KeepLoRA modules
        if use_keeplora:
            valid_keeplora_keys = {'q', 'k', 'v', 'o'}
            if not set(use_keeplora).issubset(valid_keeplora_keys):
                raise ValueError(f"use_keeplora can only contain a subset of {valid_keeplora_keys}, got {use_keeplora}")
            keeplora_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    k: KeepLoRA(in_dim=emb_dim, out_dim=emb_dim, r=lora_rank, lora_alpha=1, use_rslora=False, dtype=dtype) for k in use_keeplora
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            keeplora_list = nn.ModuleList([None] * n_layers)

        # To be optimized
        self.block_tuned = block_tuned
        self.keeplora_list = keeplora_list

        # Apply low-rank approximation to parameters
        param_names = self.cfg.t_svd_param_names
        alphas = self.cfg.t_svd_alphas
        if param_names is not None and alphas is not None:
            self.apply_low_rank_approx_to_params(clip_model.transformer, param_names, alphas)

    def apply_low_rank_approx_to_params(self, text_transformer, param_names, alphas):
        """
        Apply SVD decomposition and low-rank approximation to specified parameters.
        """
        assert len(param_names) == len(alphas), "param_names and alphas must have the same length"

        device = [int(s) for s in self.cfg.gpu_id.split(',')][0] if self.cfg.gpu_id else 'cpu'  # Handle potential empty gpu_id
        emb_dim = text_transformer.resblocks[0].attn.in_proj_weight.shape[1]  # Get emb_dim from the model

        for param_name, alpha in zip(param_names, alphas):
            for i in range(len(text_transformer.resblocks)):
                # print(f"Text Layer {i}: ", end="")
                block = text_transformer.resblocks[i]
                if param_name == 'attn.in_proj_weight':
                    # attn.in_proj_weight contains q, k, v parts
                    W = block.attn.in_proj_weight  # Shape: [3 * emb_dim, emb_dim]
                    W_q = W[:emb_dim, :]  # q part
                    W_k = W[emb_dim:2 * emb_dim, :]  # k part
                    W_v = W[2 * emb_dim:, :]  # v part

                    # Apply low-rank approximation to q, k, v separately
                    # print(f"Q: ", end="")
                    W_q_r = low_rank_approx(W_q, alpha, device)
                    # print(f"/{min(W_q.shape)}, K: ", end="")
                    W_k_r = low_rank_approx(W_k, alpha, device)
                    # print(f"/{min(W_k.shape)}, V: ", end="")
                    W_v_r = low_rank_approx(W_v, alpha, device)
                    # print(f"/{min(W_v.shape)}", end="")

                    # Merge back into attn.in_proj_weight
                    W_r = torch.cat([W_q_r, W_k_r, W_v_r], dim=0)
                    block.attn.in_proj_weight.data = W_r

                elif param_name == 'attn.out_proj.weight':
                    # print(f"O: ", end="")
                    W = block.attn.out_proj.weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.attn.out_proj.weight.data = W_r
                    # print(f"/{min(W.shape)}", end="")

                elif param_name == 'mlp.c_fc.weight':
                    # print(f"MLP_1: ", end="")
                    W = block.mlp[0].weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.mlp[0].weight.data = W_r  
                    # print(f"/{min(W.shape)}", end="")

                elif param_name == 'mlp.c_proj.weight':
                    # print(f"MLP_2: ", end="")
                    W = block.mlp[2].weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.mlp[2].weight.data = W_r
                    # print(f"/{min(W.shape)}", end="")
                else:
                    raise ValueError(f"Unsupported parameter name: {param_name}")
                # print("")


class Peft_Text(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.blocks = clip_model.transformer.resblocks
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.out_dim = clip_model.text_projection.shape[1]
        self.dtype = clip_model.dtype

    def forward(self, text: torch.Tensor, tuner: Text_Tuner=None, accumulate_mode: bool = False):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                keeplora = tuner.keeplora_list[i]
            else:
                keeplora = None

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj_weight = _mlp[0].weight
            _mlp_in_proj_bias = _mlp[0].bias
            _mlp_act = _mlp[1]
            _mlp_out_proj_weight = _mlp[2].weight
            _mlp_out_proj_bias = _mlp[2].bias

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x

            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            if keeplora is not None:
                if accumulate_mode:
                    keeplora['q'].accumulate_features(x) if 'q' in keeplora else None
                    keeplora['k'].accumulate_features(x) if 'k' in keeplora else None
                    keeplora['v'].accumulate_features(x) if 'v' in keeplora else None
                else:
                    q = q + keeplora["q"](x) if 'q' in keeplora else q
                    k = k + keeplora["k"](x) if 'k' in keeplora else k
                    v = v + keeplora["v"](x) if 'v' in keeplora else v

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            attn_mask = block.attn_mask.to(dtype=x.dtype, device=x.device) if block.attn_mask is not None else None
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)

            if keeplora is not None:
                x_hat = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
                if accumulate_mode:
                    keeplora['o'].accumulate_features(x) if 'o' in keeplora else None
                    x = x_hat
                else:
                    x = x_hat + keeplora["o"](x) if 'o' in keeplora else x_hat
            else:
                x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)

            x = x.view(_seq_len, _bsz, _emb_dim)
            x = x + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x

            x = _ln_2(x)
            
            x = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            
            x = _mlp_act(x)
            
            x = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            
            x = x + identity
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).to(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x