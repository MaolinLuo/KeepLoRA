import os.path as osp
import os
import json
import time
import copy
import datetime
from tqdm import tqdm
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

from model.customClip import CustomCLIP
from model.keeplora_helper import KeepLoRAHelper
from model.utils import AverageMeter

class Trainer:
    def __init__(self, cfg, model: CustomCLIP, device, log_txt, task_id):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.log_txt = log_txt
        self.task_id = task_id  # Only used for output and log
        if cfg.v_keeplora or cfg.t_keeplora:
            self.keeplora_helper = KeepLoRAHelper(model=self.model, v_keeplora=cfg.v_keeplora, t_keeplora=cfg.t_keeplora, prec=cfg.prec, device=self.device)

    def estimate_gradient(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        clear_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Estimates the gradients of a model's parameters over a dataset.
        """

        def record_gradients(model: torch.nn.Module, named_grads: Dict[str, torch.Tensor]) -> None:
            """
            Records the gradients of model parameters into a dictionary.
            """
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in named_grads:
                        # Initialize with the gradient from the first batch
                        named_grads[name] = param.grad.detach().clone().cpu()
                    else:
                        # Accumulate gradients from subsequent batches
                        named_grads[name] += param.grad.detach().cpu()
        cfg=self.cfg
        self.model.train()
        
        # Initialize gradient storage
        named_grads = {}
        num_batches = 0

        for param in self.model.image_encoder.parameters():
            param.requires_grad_(True)
        for param in self.model.text_encoder.parameters():
            param.requires_grad_(True)
        
        for name, param in self.model.v_tuner.named_parameters():
            param.requires_grad_(False)
        for name, param in self.model.t_tuner.named_parameters():
            param.requires_grad_(False)
        
        print(f"Starting gradient estimation on device: {device}")

        # Main estimation loop
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Estimating gradients")):
            try:
                # Clear gradients from previous iteration
                self.model.zero_grad()
                
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with autocast(device_type=self.device.type, enabled=cfg.prec == "amp"):
                    res = self.model(inputs, use_tuner=False)
                    outputs = res["outputs"]
                    loss = F.cross_entropy(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Record gradients
                record_gradients(self.model, named_grads)
                
                num_batches += 1
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
        
        # Average the gradients
        print(f"Averaging gradients over {num_batches} batches")
        for param_name in named_grads:
            named_grads[param_name] = named_grads[param_name] / num_batches

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Gradient estimation completed. Estimated gradients for {len(named_grads)} parameter groups.")
        
        return named_grads
    
    def in_proj_weight_process(self, in_proj_weight_grad, emb_dim, r_dim, q_prin_subspace, k_prin_subspace, v_prin_subspace):
        q_grad = in_proj_weight_grad[:emb_dim, :].T
        k_grad = in_proj_weight_grad[emb_dim:2*emb_dim, :].T
        v_grad = in_proj_weight_grad[2*emb_dim:, :].T

        if q_prin_subspace is not None:
            q_prin_subspace = q_prin_subspace.to(q_grad.device, dtype=q_grad.dtype)
            q_grad = q_grad - q_prin_subspace @ q_grad
        if k_prin_subspace is not None:
            k_prin_subspace = k_prin_subspace.to(k_grad.device, dtype=k_grad.dtype)
            k_grad = k_grad - k_prin_subspace @ k_grad
        if v_prin_subspace is not None:
            v_prin_subspace = v_prin_subspace.to(v_grad.device, dtype=v_grad.dtype)
            v_grad = v_grad - v_prin_subspace @ v_grad
        
        U, S, V = torch.svd_lowrank(q_grad.float(), q=4 * r_dim, niter=4)
        q_A = U[:, :r_dim]
        q_B = V.T[:r_dim, :] * S[:r_dim].unsqueeze(1)
        U, S, V = torch.svd_lowrank(k_grad.float(), q=4 * r_dim, niter=4)
        k_A = U[:, :r_dim]
        k_B = V.T[:r_dim, :] * S[:r_dim].unsqueeze(1)
        U, S, V = torch.svd_lowrank(v_grad.float(), q=4 * r_dim, niter=4)
        v_A = U[:, :r_dim]
        v_B = V.T[:r_dim, :] * S[:r_dim].unsqueeze(1)

        return q_A, q_B, k_A, k_B, v_A, v_B
    
    def out_proj_weight_process(self, out_proj_weight_grad, emb_dim, r_dim, o_prin_subspace):
        o_grad = out_proj_weight_grad[:emb_dim, :].T
        if o_prin_subspace is not None:
            o_prin_subspace = o_prin_subspace.to(o_grad.device, dtype=o_grad.dtype)
            o_grad = o_grad - o_prin_subspace @ o_grad
        
        U, S, V = torch.svd_lowrank(o_grad.float(), q=4 * r_dim, niter=4)
        o_A = U[:, :r_dim]
        o_B = V.T[:r_dim, :] * S[:r_dim].unsqueeze(1)
        
        return o_A, o_B
    
    def init_keeplora_with_grad(self, named_grads, prin_subspace_dict):
        """Initialize KeepLoRA matrices using gradients."""
        cfg = self.cfg
        
        print("Initializing KeepLoRA matrices using gradients...")

        projection_matrix_dict = dict()
        for k, v in prin_subspace_dict.items():
            projection_matrix_dict[k] = []
            for prin_subspace_layer in v:
                Uf = prin_subspace_layer @ prin_subspace_layer.T
                projection_matrix_dict[k].append(Uf)

        if cfg.v_keeplora:
            # Process vision transformer blocks
            for block_idx in range(len(self.model.v_tuner.keeplora_list)):
                module_dict = self.model.v_tuner.keeplora_list[block_idx]
                if module_dict is not None:
                    # Look for gradients for this block
                    in_proj_grad_key = f"image_encoder.blocks.{block_idx}.attn.in_proj_weight"
                    out_proj_grad_key = f"image_encoder.blocks.{block_idx}.attn.out_proj.weight"
                    
                    if in_proj_grad_key in named_grads:
                        in_proj_grad = named_grads[in_proj_grad_key]
                        emb_dim = in_proj_grad.shape[0] // 3  # in_proj_weight contains q,k,v stacked
                        r_dim = cfg.v_adapter_dim
                        
                        # Process in_proj_weight gradients (q, k, v)
                        q_A, q_B, k_A, k_B, v_A, v_B = self.in_proj_weight_process(
                            in_proj_grad, emb_dim, r_dim, 
                            projection_matrix_dict['vit_q'][block_idx] if 'vit_q' in projection_matrix_dict and len(projection_matrix_dict['vit_q']) > 0 else None,
                            projection_matrix_dict['vit_k'][block_idx] if 'vit_k' in projection_matrix_dict and len(projection_matrix_dict['vit_k']) > 0 else None,
                            projection_matrix_dict['vit_v'][block_idx] if 'vit_v' in projection_matrix_dict and len(projection_matrix_dict['vit_v']) > 0 else None
                        )
                        
                        # Initialize q, k, v LoRA matrices
                        if 'q' in module_dict and 'q' in cfg.v_keeplora:
                            module_dict['q'].lora_A.data.copy_(q_A.to(module_dict['q'].lora_A.device))
                            module_dict['q'].lora_B.data.copy_(q_B.to(module_dict['q'].lora_B.device))
                        if 'k' in module_dict and 'k' in cfg.v_keeplora:
                            module_dict['k'].lora_A.data.copy_(k_A.to(module_dict['k'].lora_A.device))
                            module_dict['k'].lora_B.data.copy_(k_B.to(module_dict['k'].lora_B.device))
                        if 'v' in module_dict and 'v' in cfg.v_keeplora:
                            module_dict['v'].lora_A.data.copy_(v_A.to(module_dict['v'].lora_A.device))
                            module_dict['v'].lora_B.data.copy_(v_B.to(module_dict['v'].lora_B.device))
                    
                    if out_proj_grad_key in named_grads:
                        out_proj_grad = named_grads[out_proj_grad_key]
                        emb_dim = out_proj_grad.shape[0]
                        r_dim = cfg.v_adapter_dim
                        
                        # Process out_proj_weight gradients (o)
                        o_A, o_B = self.out_proj_weight_process(
                            out_proj_grad, emb_dim, r_dim,
                            projection_matrix_dict['vit_o'][block_idx] if 'vit_o' in projection_matrix_dict and len(projection_matrix_dict['vit_o']) > 0 else None
                        )
                        
                        # Initialize o LoRA matrices
                        if 'o' in module_dict and 'o' in cfg.v_keeplora:
                            module_dict['o'].lora_A.data.copy_(o_A.to(module_dict['o'].lora_A.device))
                            module_dict['o'].lora_B.data.copy_(o_B.to(module_dict['o'].lora_B.device))

        if cfg.t_keeplora:
            # Process text transformer blocks
            for block_idx in range(len(self.model.t_tuner.keeplora_list)):
                module_dict = self.model.t_tuner.keeplora_list[block_idx]
                if module_dict is not None:
                    # Look for gradients for this block
                    in_proj_grad_key = f"text_encoder.blocks.{block_idx}.attn.in_proj_weight"
                    out_proj_grad_key = f"text_encoder.blocks.{block_idx}.attn.out_proj.weight"
                    
                    if in_proj_grad_key in named_grads:
                        in_proj_grad = named_grads[in_proj_grad_key]
                        emb_dim = in_proj_grad.shape[0] // 3  # in_proj_weight contains q,k,v stacked
                        r_dim = cfg.t_adapter_dim
                        
                        # Process in_proj_weight gradients (q, k, v)
                        q_A, q_B, k_A, k_B, v_A, v_B = self.in_proj_weight_process(
                            in_proj_grad, emb_dim, r_dim,
                            projection_matrix_dict['text_q'][block_idx] if 'text_q' in projection_matrix_dict and len(projection_matrix_dict['text_q']) > 0 else None,
                            projection_matrix_dict['text_k'][block_idx] if 'text_k' in projection_matrix_dict and len(projection_matrix_dict['text_k']) > 0 else None,
                            projection_matrix_dict['text_v'][block_idx] if 'text_v' in projection_matrix_dict and len(projection_matrix_dict['text_v']) > 0 else None
                        )
                        
                        # Initialize q, k, v LoRA matrices
                        if 'q' in module_dict and 'q' in cfg.t_keeplora:
                            module_dict['q'].lora_A.data.copy_(q_A.to(module_dict['q'].lora_A.device))
                            module_dict['q'].lora_B.data.copy_(q_B.to(module_dict['q'].lora_B.device))
                        if 'k' in module_dict and 'k' in cfg.t_keeplora:
                            module_dict['k'].lora_A.data.copy_(k_A.to(module_dict['k'].lora_A.device))
                            module_dict['k'].lora_B.data.copy_(k_B.to(module_dict['k'].lora_B.device))
                        if 'v' in module_dict and 'v' in cfg.t_keeplora:
                            module_dict['v'].lora_A.data.copy_(v_A.to(module_dict['v'].lora_A.device))
                            module_dict['v'].lora_B.data.copy_(v_B.to(module_dict['v'].lora_B.device))
                    
                    if out_proj_grad_key in named_grads:
                        out_proj_grad = named_grads[out_proj_grad_key]
                        emb_dim = out_proj_grad.shape[0]
                        r_dim = cfg.t_adapter_dim
                        
                        # Process out_proj_weight gradients (o)
                        o_A, o_B = self.out_proj_weight_process(
                            out_proj_grad, emb_dim, r_dim,
                            projection_matrix_dict['text_o'][block_idx] if 'text_o' in projection_matrix_dict and len(projection_matrix_dict['text_o']) > 0 else None
                        )
                        
                        # Initialize o LoRA matrices
                        if 'o' in module_dict and 'o' in cfg.t_keeplora:
                            module_dict['o'].lora_A.data.copy_(o_A.to(module_dict['o'].lora_A.device))
                            module_dict['o'].lora_B.data.copy_(o_B.to(module_dict['o'].lora_B.device))
        
        print("KeepLoRA matrix initialization using gradients complete.")

    def build_optimizer(self, num_epochs, steps_per_epoch):
        cfg = self.cfg

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        for name, param in self.model.v_tuner.named_parameters():
            param.requires_grad_(True)
        for name, param in self.model.t_tuner.named_parameters():
            param.requires_grad_(True)

        print("v_tuner param num:", sum(p.numel() for p in self.model.v_tuner.parameters()))
        print("t_tuner param num:", sum(p.numel() for p in self.model.t_tuner.parameters()))

        if cfg.classifier:
            key_str = "_".join(self.model.current_classnames).replace(".", "")
            for name, param in self.model.classifier_dict[key_str].named_parameters():
                param.requires_grad_(True)
            print("classifier param num:", sum(p.numel() for p in self.model.classifier_dict[key_str].parameters()))
            params_list = [{"params": self.model.v_tuner.parameters()},
                           {"params": self.model.t_tuner.parameters()},
                           {"params": self.model.classifier_dict[key_str].parameters()}]
        else:
            params_list = [{"params": self.model.v_tuner.parameters()},
                           {"params": self.model.t_tuner.parameters()}]

        if cfg.optim.name == 'SGD':
            self.optim = torch.optim.SGD(params_list, 
                                         lr=cfg.optim.lr,
                                         weight_decay=cfg.optim.weight_decay,
                                         momentum=0.9)
        elif cfg.optim.name == 'AdamW':
            self.optim = torch.optim.AdamW(params_list,
                                           lr=cfg.optim.lr,
                                           weight_decay=cfg.optim.weight_decay)
        else:
            raise NotImplementedError
        
        if cfg.optim.lr_scheduler == 'OneCycleLR':
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim,
                                                             max_lr=cfg.optim.lr,
                                                             epochs=num_epochs,
                                                             steps_per_epoch=steps_per_epoch)
        elif cfg.optim.lr_scheduler == 'CosineAnnealingLR':
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, num_epochs)
        else:
            raise NotImplementedError

        self.scaler = GradScaler(enabled=cfg.prec == "amp")


    def train_one_task(self, accum_loader, train_loader, num_epochs, prin_subspace_dict):
        cfg = self.cfg
        
        if cfg.v_keeplora or cfg.t_keeplora:
            self.keeplora_helper.reset_keeplora()
            gradients = self.estimate_gradient(accum_loader, device=self.device)
            self.init_keeplora_with_grad(gradients, prin_subspace_dict)
            self.model.subtract_keeplora_weights()
            self.model.zero_grad()

            torch.cuda.empty_cache()

        if cfg.classifier:
            self.model.add_classifier(accum_loader, device=self.device)

        self.build_optimizer(num_epochs=num_epochs, steps_per_epoch=len(train_loader))

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)

        self.best_epoch = -1
        self.best_acc = -1
        
        for epoch_idx in range(num_epochs):
            self.model.train()
            end = time.time()
            
            num_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                data_time.update(time.time() - end)
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with autocast(device_type=self.device.type, enabled=cfg.prec == "amp"):
                    res = self.model(inputs)
                    outputs = res["outputs"]
                    loss = F.cross_entropy(outputs, targets)

                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                
                with torch.no_grad():
                    pred = outputs.argmax(dim=1)
                    correct = pred.eq(targets).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)
                
                if batch_idx % 10 == 0:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))
                    self.log_txt(" ".join(info))
                
                end = time.time()
            
            log = f"Task {self.task_id + 1} Epoch {epoch_idx + 1} "
            log += f"Train acc: {acc_meter.avg:.2f} "
            log += f"Loss: {loss_meter.avg:.4f}"
            self.log_txt(log)

        torch.cuda.empty_cache()

    def evaluate_one(self, loader, metric_logger, task_id):
        cfg = self.cfg
        right_num = 0
        sample_num = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with autocast(device_type=self.device.type, enabled=cfg.prec == "amp"):
                    # Do not use tuner for evaluation because it has been merged to backbone
                    res = self.model(inputs, use_tuner=False)
                    outputs = res["outputs"]

                task_ids = torch.IntTensor([task_id]).repeat(inputs.size(0))
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                right_num += torch.sum(outputs.argmax(dim=1) == targets).item()
                sample_num += inputs.size(0)

        return right_num / sample_num

    def save_model(self):
        cfg = self.cfg
        with torch.no_grad():
            self.model.eval()
            model = copy.deepcopy(self.model)
            model.merge_keeplora_weights()
        save_dict = {
            'image_encoder': model.image_encoder.state_dict(),
            'text_encoder': model.text_encoder.state_dict(),
        }
        save_dir = os.path.join(cfg.log_path, 'ckpt')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, os.path.join(save_dir, f'task_{self.task_id}.pt'))

        torch.cuda.empty_cache()
    
    def load_model(self):
        cfg = self.cfg
        load_file = os.path.join(cfg.log_path, 'ckpt', f'task_{self.task_id}.pt')
        if not osp.exists(load_file):
            raise FileNotFoundError('Model not found at "{}"'.format(load_file))

        state_dict = torch.load(load_file, map_location="cpu")
        print(f"Loading backbone weights from {load_file}")
        self.model.image_encoder.load_state_dict(state_dict['image_encoder'])
        self.model.text_encoder.load_state_dict(state_dict['text_encoder'])