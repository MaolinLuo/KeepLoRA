import argparse
import os
import sys
import torch
from torchvision import transforms

from MTIL_datasets.collections import *
from model.utils import get_transform, set_random_seed
from model.setup_cfg import setup_cfg, print_args
from trainer.framework4MTIL import Framework4MTIL



def run_exp(cfg):
    train_dataset = []
    test_dataset = []
    classes_names = []
    templates = []

    train_transforms = get_transform(cfg)
    test_transforms = get_transform(cfg)

    dataset_classes = {
        'Aircraft': Aircraft, 'Caltech101': Caltech101, 'CIFAR100': CIFAR100,
        'DTD': DTD, 'EuroSAT': EuroSAT, 'Flowers': Flowers, 'Food': Food, 'MNIST': MNIST,
        'OxfordPet': OxfordPet, 'StanfordCars': StanfordCars, 'SUN397': SUN397, 'Places': Places, 'ImageNet': ImageNet
    }

    for dataset_name in cfg.tasks:
        dataset = dataset_classes[dataset_name](
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            location=cfg.dataset_root,
            batch_size=cfg.optim.batch_size
        )
        train_dataset.append(dataset.train_dataset)
        test_dataset.append(dataset.test_dataset)
        classes_names.append(dataset.classnames)
        templates.append(dataset.templates[0])

    trainer = Framework4MTIL(cfg, classes_names, templates)

    datasets = {'train': train_dataset, 'test': test_dataset}
    trainer.train_and_evaluate(datasets)


def main(args):
    cfg = setup_cfg(args)
    cfg.command = ' '.join(sys.argv)

    cfg.log_path = os.path.join('output', 'experiments', f'{cfg.dataset}', cfg.add_info)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    with open(os.path.join(cfg.log_path, 'config.yaml'), 'w') as f: 
        f.write(cfg.dump())
    
    print_args(args, cfg)

    set_random_seed(cfg.seed)
    run_exp(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/keeplora_order1.yaml", help="path to config")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)