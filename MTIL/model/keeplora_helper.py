from tqdm import tqdm

import torch
from torch.amp import autocast

from model.customClip import CustomCLIP

class KeepLoRAHelper:
    def __init__(self, model: CustomCLIP, v_keeplora, t_keeplora, prec, device):
        self.model = model
        self.v_keeplora = v_keeplora
        self.t_keeplora = t_keeplora
        self.prec = prec
        self.device = device

    def reset_keeplora(self):
        """Reset KeepLoRA parameters."""

        def _reset_keeplora(keeplora_list):
            for module_dict in keeplora_list:
                if module_dict is not None:
                    for i in ('q', 'k', 'v', 'o'):
                        if i in module_dict:
                            module_dict[i].reset_parameters(device=self.device)
        
        if self.v_keeplora:
            _reset_keeplora(self.model.v_tuner.keeplora_list)
        if self.t_keeplora:
            _reset_keeplora(self.model.t_tuner.keeplora_list)
        print("KeepLoRA reset complete.")

    def release_keeplora_buffer(self):
        """Release buffers for KeepLoRA parameters."""

        def _release_keeplora_buffer(keeplora_list):
            for module_dict in keeplora_list:
                if module_dict is not None:
                    for i in ('q', 'k', 'v', 'o'):
                        if i in module_dict:
                            module_dict[i].release_buffer()

        if self.v_keeplora:
            _release_keeplora_buffer(self.model.v_tuner.keeplora_list)
        if self.t_keeplora:
            _release_keeplora_buffer(self.model.t_tuner.keeplora_list)

    def accumulate_features_for_task(self, loader):
        """Runs a pass over the data to accumulate features for KeepLoRA."""
        print(f"Accumulating features for KeepLoRA...")

        self.model.eval()

        peft_vit = self.model.image_encoder
        v_tuner = self.model.v_tuner
        peft_text = self.model.text_encoder
        t_tuner = self.model.t_tuner
        tokenized_prompts = self.model.prompt_processor()

        with torch.no_grad():
            if self.v_keeplora:
                for batch in tqdm(loader, desc="Accumulating Features"):
                    inputs = batch[0]
                    inputs = inputs.to(self.device)
                    with autocast(device_type=self.device.type, enabled=self.prec == "amp"):
                        peft_vit(inputs, tuner=v_tuner, accumulate_mode=True)
            if self.t_keeplora:
                with autocast(device_type=self.device.type, enabled=self.prec == "amp"):
                    peft_text(tokenized_prompts, tuner=t_tuner, accumulate_mode=True)

    def initialize_keeplora_for_task(self, prin_subspace_dict):
        """Initializes KeepLoRA matrices using SVD after feature accumulation."""

        projection_matrix_dict = dict()
        for k, v in prin_subspace_dict.items():
            projection_matrix_dict[k] = []
            for prin_subspace_layer in v:
                Uf = prin_subspace_layer @ prin_subspace_layer.T
                projection_matrix_dict[k].append(Uf)

        vit_map = {'q': 'vit_q', 'k': 'vit_k', 'v': 'vit_v', 'o': 'vit_o'}
        text_map = {'q': 'text_q', 'k': 'text_k', 'v': 'text_v', 'o': 'text_o'}

        def _initialize_keeplora(keeplora_list, map_dict):
            for i, module_dict in enumerate(keeplora_list):
                if module_dict is not None:
                    for k, v in map_dict.items():
                        if k in module_dict:
                            module_dict[k].initialize_lora_matrices(projection_matrix=projection_matrix_dict[v][i] if len(projection_matrix_dict[v]) > 0 else None)
                
        if self.v_keeplora:
            _initialize_keeplora(self.model.v_tuner.keeplora_list, vit_map)
        if self.t_keeplora:
            _initialize_keeplora(self.model.t_tuner.keeplora_list, text_map)
        print("KeepLoRA initialization complete.")

