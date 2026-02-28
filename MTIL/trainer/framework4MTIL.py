import os
import gc
import json
import time
import copy
import datetime
from tqdm import tqdm
from continuum.metrics import Logger

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.customClip import CustomCLIP, load_clip_to_cpu
from model.utils import cal_MTIL_metrics
from model.prin_subspace import PrinSubspace
from model import prin_subspace_helper

from trainer.trainer import Trainer


# A framework of continual learning in MTIL setting
class Framework4MTIL:
    def __init__(self, cfg, classes_names, templates):
        self.cfg = cfg
        self.metric_logger = Logger()
        self.metric_writer_path = os.path.join(cfg.log_path, 'metrics.json')
        self.txt_writer_path = os.path.join(cfg.log_path, 'output.txt')
        self._init_logger()
        self.model, self.device = self.build_model(classes_names, templates)
        if not cfg.zero_shot:
            self.prin_subspace_name_list, self.threshold_dict, self.threshold_dict2 = prin_subspace_helper.init_prin_subspace_helper(cfg)
            self.prin_subspace = PrinSubspace(prin_subspace_name_list=self.prin_subspace_name_list, log_txt=self.log_txt)

    def _init_logger(self):
        with open(self.metric_writer_path, 'w') as f:
            pass
        with open(self.txt_writer_path, 'w') as f:
            pass

    def log_writer(self, log):
        with open(self.metric_writer_path, 'a') as f:
            f.write(json.dumps(log) + '\n')

    def log_txt(self, log):
        with open(self.txt_writer_path, 'a') as f:
            f.write(log + '\n')

    def update_classnames(self, task_id):
        self.model.update_classnames(task_id)

    def build_dataloader(self, dataset, task_id, shuffle, drop_last=False):
        cfg = self.cfg
        batch_size = cfg.optim.batch_size
        if isinstance(dataset, list):
            return DataLoader(dataset[task_id], batch_size=int(batch_size), shuffle=shuffle, num_workers=8, drop_last=drop_last)
        else:
            return None

    def build_model(self, classes_names, templates):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.model_backbone_name})")
        clip_model = load_clip_to_cpu(cfg, cfg.prec)
        print("Building custom CLIP")
        model = CustomCLIP(cfg, classes_names, templates, clip_model, 
                           classifier=cfg.classifier if not cfg.zero_shot else None)

        device_id = [int(s) for s in cfg.gpu_id.split(',')]
        device = torch.device(f"cuda:{device_id[0]}" if torch.cuda.is_available() else "cpu")
        if len(device_id) > 1:
            model.to(device)
            model = torch.nn.DataParallel(model, device_ids=device_id)
        else:
            model.to(device)
        return model, device

    def save_model(self, trainer: Trainer):
        trainer.save_model()

    def load_model(self, trainer: Trainer):
        trainer.load_model()

    def train_and_evaluate(self, datasets):
        cfg = self.cfg
        acc_list = []

        if cfg.zero_shot:
            with torch.no_grad():
                for task_id in tqdm(range(len(datasets['test']))):
                    self.update_classnames(task_id)
                    eval_loader = self.build_dataloader(datasets['test'], task_id, shuffle=False)
                    for batch in eval_loader:
                        inputs, targets = batch[0], batch[1]
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        res = self.model(inputs)
                        outputs = res["outputs"]

                        task_ids = torch.IntTensor([task_id]).repeat(inputs.size(0))
                        self.metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

                cur_all_task_acc = self.metric_logger.accuracy_per_task
                acc_list.append(cur_all_task_acc)
                log = {'acc_per_task': [round(100 * float(acc_t), 2) for acc_t in cur_all_task_acc]}
                self.log_writer(log)
                print(log)
                return

        if cfg.eval_only:
            self.evaluate_all(datasets, acc_list)
            return

        for task_id in range(len(datasets['train'])):
            print(f"Training for task {task_id + 1} has started.")
            self.update_classnames(task_id)

            trainer = Trainer(cfg=cfg, model=self.model, device=self.device, log_txt=self.log_txt, task_id=task_id)

            train_dataset, _ = datasets['train'], datasets['test']
            accum_loader = self.build_dataloader(train_dataset, task_id, shuffle=True, drop_last=True)
            train_loader = self.build_dataloader(train_dataset, task_id, shuffle=True)

            if task_id == 0 and any(abs(value) > 1e-8 for value in self.threshold_dict2.values()):
                feature_matrix = prin_subspace_helper.get_accumulated_weight_matrix_list(self.model, self.prin_subspace_name_list)
                self.prin_subspace.update_prin_subspace(mat_list_dict=feature_matrix,
                                    threshold_dict=self.threshold_dict2)

            epoch = cfg.optim.epoch_list[task_id]

            trainer.train_one_task(accum_loader, train_loader, epoch, self.prin_subspace.prin_subspace_dict)
            self.save_model(trainer)    # merge and save
            self.load_model(trainer)

            if cfg.v_keeplora or cfg.t_keeplora:
                trainer.keeplora_helper.reset_keeplora()
                trainer.keeplora_helper.accumulate_features_for_task(accum_loader)
                feature_matrix = prin_subspace_helper.get_accumulated_feature_matrix_list(self.model, self.prin_subspace_name_list)
                self.prin_subspace.update_prin_subspace(mat_list_dict=feature_matrix,
                                    threshold_dict=self.threshold_dict)

            print(f"Evaluation for task {task_id + 1} has started.")
            self.evaluate_all(trainer, datasets, acc_list)

            del trainer, accum_loader, train_loader, feature_matrix
            gc.collect()
            torch.cuda.empty_cache()
            
        res = cal_MTIL_metrics(acc_list)
        self.log_writer(res["transfer"])
        self.log_writer(res["avg"])
        self.log_writer(res["last"])
        self.log_writer(res["results_mean"])


    def evaluate_all(self, trainer: Trainer, datasets, acc_list):
        eval_dataset = datasets['test']
        self.model.eval()

        for task_id in tqdm(range(len(eval_dataset))):
            self.update_classnames(task_id)
            eval_loader = self.build_dataloader(eval_dataset, task_id, shuffle=False)
            trainer.evaluate_one(eval_loader, self.metric_logger, task_id)

        cur_all_task_acc = self.metric_logger.accuracy_per_task
        acc_list.append(cur_all_task_acc)
        log = {'acc_per_task': [round(100 * float(acc_t), 2) for acc_t in cur_all_task_acc]}
        self.log_writer(log)
        print(log)
        self.metric_logger.end_task()
