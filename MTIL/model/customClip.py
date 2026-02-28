import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from clip import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.peft_text import Peft_Text, Text_Tuner
from model.peft_vit import Peft_ViT, ViT_Tuner
from model.classifiers import LinearClassifier, CosineClassifier


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, prec="fp16"):
    backbone_name = cfg.model_backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


# tokenizer processor
class TokenizerProcessor(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model):
        super().__init__()

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_size[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if isinstance(classnames[0], list):
            self.n_cls = 0
            self.class_ids_per_task = []
            self.classnames = []
            for idx, cls_name in enumerate(classnames):
                cur_n = len(cls_name)
                self.class_ids_per_task.append([i for i in range(self.n_cls, self.n_cls+cur_n)])
                cls_name = [templates[idx](name) for name in cls_name]
                self.classnames += cls_name
                self.n_cls += cur_n
        else:
            raise NotImplementedError
        self.cur_n_cls = 0

        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.all_name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        all_prompts = [name for name in self.classnames]
        self.register_buffer("all_tokenized_prompts", torch.cat([clip.tokenize(p) for p in all_prompts]))
        with torch.no_grad():
            self.register_buffer("all_embedding", clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype))
            self.register_buffer("tokenized_prompts", self.all_tokenized_prompts.clone())

    def forward(self):
        return self.tokenized_prompts
    
    def update_classnames(self, task_id):
        class_idx = self.class_ids_per_task[task_id]
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.int, device=self.all_embedding.device)
        self.tokenized_prompts = self.all_tokenized_prompts[class_idx_tensor]
        self.name_lens = [self.all_name_lens[idx] for idx in class_idx]
        self.cur_n_cls = len(class_idx)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model:CLIP, classifier=None):
        super().__init__()
        self.prompt_processor = TokenizerProcessor(cfg, classnames, templates, clip_model)
        self.image_encoder = Peft_ViT(clip_model.visual)
        self.text_encoder = Peft_Text(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = clip_model.logit_scale.device
        self.classnames = classnames

        self.v_tuner = ViT_Tuner(cfg, clip_model.visual)
        self.t_tuner = Text_Tuner(cfg, clip_model)
        
        self.classifier = classifier
        
        if self.classifier:
            self.classifier_dict = nn.ModuleDict()
            self.feat_dim = clip_model.visual.output_dim
    
    def add_classifier(self, dataloader, device):
        num_classes=len(self.current_classnames)
        classifier = eval(self.classifier)(
            feat_dim=self.feat_dim,
            num_classes=num_classes,
            dtype=self.dtype
        )
        classifier = classifier.to(device)
        self.init_head_class_mean(dataloader, classifier, num_classes, device)
        key_str = "_".join(self.current_classnames).replace(".", "")
        self.classifier_dict[key_str] = classifier

    @torch.no_grad()
    def init_head_class_mean(self, dataloader, classifier, num_classes, device):
        all_features = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Initialize head with class means"):
            image = batch[0]
            label = batch[1]

            image = image.to(device)
            label = label.to(device)

            output = self.forward(image)
            feature = output["image_features"]

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        classifier.apply_weight(class_means)

    def forward(self, image, use_tuner=True):
        res = {}

        tokenized_prompts = self.prompt_processor()  # [bs*n_cls, 77, ctx_dim]
        text_features = self.text_encoder(tokenized_prompts, self.t_tuner if use_tuner else None)  # [bs*n_cls, model_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image, self.v_tuner if use_tuner else None)  # [bs, model_dim]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()  # [bs, n_cls]

        key_str = "_".join(self.current_classnames).replace(".", "")
        if self.classifier and key_str in self.classifier_dict:
            classifier = self.classifier_dict[key_str]
            logits += classifier(image_features)

        res["outputs"] = logits
        res["image_features"] = image_features
        
        return res

    def update_classnames(self, task_id):
        self.prompt_processor.update_classnames(task_id)
        self.current_classnames = self.classnames[task_id]
    
    def merge_keeplora_weights(self):
        """Merges the learned KeepLoRA delta weights into the backbone for both v_tuner and t_tuner."""

        print(f"Merging KeepLoRA weights into backbone...")
        peft_vit = self.image_encoder
        v_tuner = self.v_tuner
        peft_text = self.text_encoder
        t_tuner = self.t_tuner

        with torch.no_grad():
            # Vision (ViT) KeepLoRA
            for i, module_dict in enumerate(v_tuner.keeplora_list):
                if module_dict is not None:
                    delta_W_q = module_dict['q'].get_delta_weight() if 'q' in module_dict else None
                    delta_W_k = module_dict['k'].get_delta_weight() if 'k' in module_dict else None
                    delta_W_v = module_dict['v'].get_delta_weight() if 'v' in module_dict else None

                    attn = peft_vit.blocks[i].attn
                    W_qkv = attn.in_proj_weight.data
                    W_q, W_k, W_v = W_qkv.chunk(3, dim=0)

                    W_q = W_q + delta_W_q.to(W_q.device, dtype=W_q.dtype) if delta_W_q is not None else W_q
                    W_k = W_k + delta_W_k.to(W_k.device, dtype=W_k.dtype) if delta_W_k is not None else W_k
                    W_v = W_v + delta_W_v.to(W_v.device, dtype=W_v.dtype) if delta_W_v is not None else W_v

                    W_qkv_new = torch.cat([W_q, W_k, W_v], dim=0)
                    attn.in_proj_weight.data.copy_(W_qkv_new)
                    
                    delta_W_o = module_dict['o'].get_delta_weight() if 'o' in module_dict else None
                    
                    W_o = attn.out_proj.weight
                    
                    W_o = W_o + delta_W_o.to(W_o.device, dtype=W_o.dtype) if delta_W_o is not None else W_o
                    attn.out_proj.weight.data.copy_(W_o)
                    
            # Text KeepLoRA
            for i, module_dict in enumerate(t_tuner.keeplora_list):
                if module_dict is not None:
                    delta_W_q = module_dict['q'].get_delta_weight() if 'q' in module_dict else None
                    delta_W_k = module_dict['k'].get_delta_weight() if 'k' in module_dict else None
                    delta_W_v = module_dict['v'].get_delta_weight() if 'v' in module_dict else None

                    attn = peft_text.blocks[i].attn
                    W_qkv = attn.in_proj_weight.data
                    W_q, W_k, W_v = W_qkv.chunk(3, dim=0)

                    W_q = W_q + delta_W_q.to(W_q.device, dtype=W_q.dtype) if delta_W_q is not None else W_q
                    W_k = W_k + delta_W_k.to(W_k.device, dtype=W_k.dtype) if delta_W_k is not None else W_k
                    W_v = W_v + delta_W_v.to(W_v.device, dtype=W_v.dtype) if delta_W_v is not None else W_v

                    W_qkv_new = torch.cat([W_q, W_k, W_v], dim=0)
                    attn.in_proj_weight.data.copy_(W_qkv_new)
                    
                    delta_W_o = module_dict['o'].get_delta_weight() if 'o' in module_dict else None
                    
                    W_o = attn.out_proj.weight
                    
                    W_o = W_o + delta_W_o.to(W_o.device, dtype=W_o.dtype) if delta_W_o is not None else W_o
                    attn.out_proj.weight.data.copy_(W_o)
        print("KeepLoRA weight merging complete.")

    def subtract_keeplora_weights(self):
        peft_vit = self.image_encoder
        v_tuner = self.v_tuner
        peft_text = self.text_encoder
        t_tuner = self.t_tuner

        with torch.no_grad():
            # Vision (ViT) KeepLoRA
            for i, module_dict in enumerate(v_tuner.keeplora_list):
                if module_dict is not None:
                    delta_W_q = module_dict['q'].get_delta_weight() if 'q' in module_dict else None
                    delta_W_k = module_dict['k'].get_delta_weight() if 'k' in module_dict else None
                    delta_W_v = module_dict['v'].get_delta_weight() if 'v' in module_dict else None

                    attn = peft_vit.blocks[i].attn
                    W_qkv = attn.in_proj_weight.data
                    W_q, W_k, W_v = W_qkv.chunk(3, dim=0)

                    W_q = W_q - delta_W_q.to(W_q.device, dtype=W_q.dtype) if delta_W_q is not None else W_q
                    W_k = W_k - delta_W_k.to(W_k.device, dtype=W_k.dtype) if delta_W_k is not None else W_k
                    W_v = W_v - delta_W_v.to(W_v.device, dtype=W_v.dtype) if delta_W_v is not None else W_v

                    W_qkv_new = torch.cat([W_q, W_k, W_v], dim=0)
                    attn.in_proj_weight.data.copy_(W_qkv_new)
                    
                    delta_W_o = module_dict['o'].get_delta_weight() if 'o' in module_dict else None
                    
                    W_o = attn.out_proj.weight
                    
                    W_o = W_o - delta_W_o.to(W_o.device, dtype=W_o.dtype) if delta_W_o is not None else W_o
                    attn.out_proj.weight.data.copy_(W_o)
                    
            # Text KeepLoRA
            for i, module_dict in enumerate(t_tuner.keeplora_list):
                if module_dict is not None:
                    delta_W_q = module_dict['q'].get_delta_weight() if 'q' in module_dict else None
                    delta_W_k = module_dict['k'].get_delta_weight() if 'k' in module_dict else None
                    delta_W_v = module_dict['v'].get_delta_weight() if 'v' in module_dict else None

                    attn = peft_text.blocks[i].attn
                    W_qkv = attn.in_proj_weight.data
                    W_q, W_k, W_v = W_qkv.chunk(3, dim=0)

                    W_q = W_q - delta_W_q.to(W_q.device, dtype=W_q.dtype) if delta_W_q is not None else W_q
                    W_k = W_k - delta_W_k.to(W_k.device, dtype=W_k.dtype) if delta_W_k is not None else W_k
                    W_v = W_v - delta_W_v.to(W_v.device, dtype=W_v.dtype) if delta_W_v is not None else W_v

                    W_qkv_new = torch.cat([W_q, W_k, W_v], dim=0)
                    attn.in_proj_weight.data.copy_(W_qkv_new)
                    
                    delta_W_o = module_dict['o'].get_delta_weight() if 'o' in module_dict else None
                    
                    W_o = attn.out_proj.weight
                    
                    W_o = W_o - delta_W_o.to(W_o.device, dtype=W_o.dtype) if delta_W_o is not None else W_o
                    attn.out_proj.weight.data.copy_(W_o)