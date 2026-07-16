# KeepLoRA++: Continual Learning with Layer-Scaled Residual Gradient Adaptation

This repository provides the official implementations of KeepLoRA and KeepLoRA++:

- **`v1`** contains the code for [KeepLoRA: Continual Learning with Residual Gradient Adaptation](https://openreview.net/forum?id=T3Vc5fkTzV), accepted at **ICLR 2026**.
- **`v2`** contains the code for [KeepLoRA++: Continual Learning with Layer-Scaled Residual Gradient Adaptation](https://arxiv.org/abs/2606.16256), which extends KeepLoRA and evaluates continual learning on image classification, visual question answering, and video understanding.

## Hardware

| Task | Hardware |
| --- | --- |
| Image classification (MTIL) | 1 x NVIDIA RTX 4090 (24 GB) |
| Visual question answering (MLLM-DCL and UCIT) | 4 x NVIDIA H100 (80 GB each); at least 2 x H100 are supported |
| Video understanding (CL-VISTA) | 2 x NVIDIA H100 (80 GB each) |

## Experiments on MTIL Benchmark

### Environment

Create an environment and install dependencies:

```bash
cd ./MTIL
conda create -n keeplora_mtil python=3.11.11
conda activate keeplora_mtil
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Model

The pre-trained CLIP model will be automatically downloaded.

### Dataset preparation


The dataset is organized according to [ZSCL](https://github.com/Thunderbeee/ZSCL). If you are looking for a source to download the raw datasets, you can refer to [https://www.modelscope.cn/datasets/ForestLuo/X-TAIL](https://www.modelscope.cn/datasets/ForestLuo/X-TAIL).

Put files in the following locations and update the `dataset_root` path in the data configuration files [keeplora_order1.yaml](MTIL/configs/keeplora_order1.yaml), [keeplora_order2.yaml](MTIL/configs/keeplora_order2.yaml), etc.

```sh
/your_dataset_path/MTIL
 ├─ caltech101
 │  └─ 101_ObjectCategories
 ├─ cifar-100-python
 │  ├─ meta
 │  ├─ test
 │  └─ train
 ├─ dtd/dtd
 │  ├─ images
 │  ├─ imbd
 │  └─ labels
 ├─ eurosat
 │  └─ 2750
 ├─ fgvc-aircraft-2013b/data
 │  ├─ images
 │  ├─ families.txt
 │  ├─ ...
 │  └─ variants.txt
 ├─ flowers-102
 │  ├─ jpg
 │  ├─ imagelabels.mat
 │  └─ setid.mat
 ├─ food-101
 │  ├─ images
 │  └─ meta
 ├─ MNIST/raw
 │  ├─ t10k-images-idx3-ubyte
 │  ├─ t10k-labels-idx1-ubyte
 │  ├─ train-images-idx3-ubyte
 │  └─ train-labels-idx1-ubyte
 ├─ oxford-iiit-pet
 │  ├─ annotations
 │  └─ images
 ├─ stanford_cars
 │  ├─ cars_test
 │  ├─ cars_train
 │  ├─ devkit
 │  └─ cars_test_annos_withlabels.mat
 └─ SUN397
    ├─ a
    ├─ ...
    ├─ y
    └─ ClassName.txt
```

### Reproduction

To reproduce the main result in the paper, please run:

```bash
# run KeepLoRA++ on order-I setting
python main.py --config-path configs/keeplora_order1.yaml

# run KeepLoRA++ on order-II setting
python main.py --config-path configs/keeplora_order2.yaml

# run KeepLoRA++cls on order-I setting
python main.py --config-path configs/keeplora+_order1.yaml

# run KeepLoRA++cls on order-II setting
python main.py --config-path configs/keeplora+_order2.yaml
```

## Experiments on MLLM-DCL and UCIT Benchmarks

### Environment

Create an environment and install dependencies:

```bash
cd ./DCL_UCIT/KeepLoRA
conda create -n keeplora_vqa python=3.10 -y
conda activate keeplora_vqa
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e ".[train]"
```

For installing [flash-attn](https://github.com/Dao-AILab/flash-attention/releases), we recommend downloading version 2.6.3 from the official repository according to your CUDA and PyTorch versions, and placing it in a local directory for manual installation. For example:

```bash
pip install flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Model

Please download the [LLaVA-1.5-7B](https://arxiv.org/pdf/2310.03744) model to your local directory.

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir /your_model_path/llava-v1.5-7b
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /your_model_path/clip-vit-large-patch14-336
```

### Dataset preparation


The dataset is organized according to [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [MLLM-DCL](https://github.com/bjzhb666/MLLM-CL).

Put files in the following locations and update the configs.

```sh
/your_dataset_path/DCL_UCIT
 ├─ Domain_data
 │  ├─ AD
 │  ├─ Med
 │  ├─ RS
 │  ├─ Sci
 │  └─ Fin
 └─ UCIT
    ├─ datasets
    ├─ ArxivQA
    ├─ CLEVR-Math
    ├─ Flickr30k
    ├─ IconQA
    ├─ ImageNet-R
    └─ VizWiz
```

### Reproduction

To reproduce the main result in the paper, please run:

```bash
# run KeepLoRA++ on MLLM-DCL
bash scripts/Train_DCL/train_all.sh

# run KeepLoRA++ on UCIT setting
bash scripts/Train_UCIT/train_all.sh
```

## Experiments on CL-VISTA Benchmark

The video-understanding implementation is based on Video-LLaVA and follows the setup of [MCITlib](https://github.com/Ghy0501/MCITlib). Please refer to that repository for optional benchmarks and more detailed preparation instructions.

### Environment

Create an environment using the original Video-LLaVA setup:

```bash
cd ./CL_VISTA/KeepLoRA
conda create -n keeplora_vista python=3.10 -y
conda activate keeplora_vista
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

Please install a compatible FlashAttention build for your CUDA and PyTorch versions before training.

The evaluation driver uses the `keeplora_vista` environment for inference and a separate `transformers` environment for the local judge model. If your judge environment has a different name, update `EVAL_ENV` in [`Eval_CVU.sh`](CL_VISTA/KeepLoRA/scripts/MCITlib/Eval/Eval_CVU.sh).

### Model

Download Video-LLaVA, its video tower, the CLIP text tower, and the judge model:

```bash
huggingface-cli download LanguageBind/Video-LLaVA-7B --local-dir /your_model_path/Video-LLaVA-7B
huggingface-cli download LanguageBind/LanguageBind_Video_merge --local-dir /your_model_path/LanguageBind_Video_merge
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /your_model_path/clip-vit-large-patch14-336
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir /your_model_path/Qwen3-30B-A3B-Instruct-2507
```

Update [`videollava.json`](CL_VISTA/configs/model_configs/videollava.json) and the model and judge paths in the training and evaluation configs. The files under [`CL_VISTA/examples/Video-LLaVA-7B`](CL_VISTA/examples/Video-LLaVA-7B) show the required changes to the downloaded model's `config.json` and `generation_config.json` and should be kept as references.

### Dataset preparation

Download [CL-VISTA](https://huggingface.co/datasets/MLLM-CL/CL-VISTA), organize it as follows, and update the paths under [`CL_VISTA/configs/data_configs/CL-VISTA`](CL_VISTA/configs/data_configs/CL-VISTA):

```sh
/your_dataset_path/CL-VISTA
 ├─ Counting
 ├─ GUI
 ├─ Movie
 ├─ Science
 ├─ Space
 ├─ Sports
 ├─ STAR
 ├─ Traffic
 └─ train_VISTA_joint.json
```

The `Space` split requires the additional ScanNet preparation described in the [MCITlib README](https://github.com/Ghy0501/MCITlib#benchmarks).

### Reproduction

Replace `/your_path/MCITlib_v3` in the CL-VISTA shell scripts with the absolute path to `CL_VISTA`, and update `/your_conda_path`, `/your_model_path`, `/your_data_path`, and the checkpoint paths for your machine. Then run from `CL_VISTA/KeepLoRA`:

```bash
bash scripts/MCITlib/Train/train_CVU.sh
```

## Citation

If you find this repository useful for your work, please consider citing the corresponding paper.

```bibtex
@article{luo2026keeplora++,
   title={KeepLoRA++: Continual Learning with Layer-Scaled Residual Gradient Adaptation},
   author={Luo, Mao-Lin and Zhang, Yi-Lin and Zhou, Zi-Hao and Hong, Yankun and Tong, Xialiang and Yuan, Mingxuan and Wei, Tong and Zhang, Min-Ling},
   journal={arXiv preprint arXiv:2606.16256},
   year={2026}
}

@inproceedings{luo2026keeplora,
	title={KeepLo{RA}: Continual Learning with Residual Gradient Adaptation},
	author={Mao-Lin Luo and Zi-Hao Zhou and Yi-Lin Zhang and Yuanyu Wan and Min-Ling Zhang and Tong Wei},
	booktitle={The Fourteenth International Conference on Learning Representations},
	year={2026},
	url={https://openreview.net/forum?id=T3Vc5fkTzV}
}
```

## Acknowledgment

We thank the authors of the following repositories for code reference: [[ZSCL]](https://github.com/Thunderbeee/ZSCL), [[InfLoRA]](https://github.com/liangyanshuo/InfLoRA), [[MCITlib]](https://github.com/Ghy0501/MCITlib).

