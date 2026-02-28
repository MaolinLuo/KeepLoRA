# KeepLoRA: Continual Learning with Residual Gradient Adaptation

This is the source code for our paper [KeepLoRA: Continual Learning with Residual Gradient Adaptation](https://openreview.net/forum?id=T3Vc5fkTzV) which has been accepted to **ICLR 2026**.

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

### Hardware

The experiments can be reproduced using a single NVIDIA 4090 GPU with 24GB of memory.

### Model

The pre-trained CLIP model will be automatically downloaded.

### Dataset preparation


The dataset is organized according to [ZSCL](https://github.com/Thunderbeee/ZSCL). If you are looking for a source to download the raw datasets, you can refer to [https://www.modelscope.cn/datasets/ForestLuo/X-TAIL](https://www.modelscope.cn/datasets/ForestLuo/X-TAIL).

Put files in the following locations and update the `dataset_root` path in the data configure files [keeplora_order1.yaml](configs/keeplora_order1.yaml), [keeplora_order2.yaml](configs/keeplora_order2.yaml), etc.

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
# run KeepLoRA on order-I setting
python main.py --config-path configs/keeplora_order1.yaml

# run KeepLoRA on order-II setting
python main.py --config-path configs/keeplora_order2.yaml

# run KeepLoRA+ on order-I setting
python main.py --config-path configs/keeplora+_order1.yaml

# run KeepLoRA+ on order-II setting
python main.py --config-path configs/keeplora+_order2.yaml
```

## Experiments on MLLM-DCL and UCIT Benchmarks

### Environment

Create an environment and install dependencies:

```bash
cd ./MCITlib/KeepLoRA
conda create -n MCITlib python=3.10 -y
conda activate MCITlib
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e ".[train]"
```

For installing [flash-attn](https://github.com/Dao-AILab/flash-attention/releases), we recommend downloading version 2.6.3 from the official repository according to your CUDA and PyTorch versions, and placing it in a local directory for manual installation. For example:

```bash
pip install flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Hardware

The experiments can be reproduced using 4 $\times$ NVIDIA H100 GPU with 80GB of memory.

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
/your_dataset_path/MCITlib
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
# run KeepLoRA on MLLM-DCL
bash scripts/Train_DCL/train_all.sh

# run KeepLoRA on UCIT setting
bash scripts/Train_UCIT/train_all.sh
```



## Citation

If you find this repository useful for your work, please consider citing our paper:

```bibtex
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

