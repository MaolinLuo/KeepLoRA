#!/bin/bash

set -e

bash scripts/Train_UCIT/extract_weights.sh configs/model_configs/LLaVA/UCIT/train_pre/task0.json 0.6

# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task1.json configs/data_configs/UCIT/ImageNet-R.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task1.sh configs/model_configs/LLaVA/UCIT/train/task1.json configs/data_configs/UCIT/ImageNet-R.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task1.json configs/data_configs/UCIT/ImageNet-R.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 1
# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task2.json configs/data_configs/UCIT/ArxivQA.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task2.sh configs/model_configs/LLaVA/UCIT/train/task2.json configs/data_configs/UCIT/ArxivQA.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task2.json configs/data_configs/UCIT/ArxivQA.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 2
# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task3.json configs/data_configs/UCIT/VizWiz.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task3.sh configs/model_configs/LLaVA/UCIT/train/task3.json configs/data_configs/UCIT/VizWiz.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task3.json configs/data_configs/UCIT/VizWiz.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 3
# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task4.json configs/data_configs/UCIT/IconQA.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task4.sh configs/model_configs/LLaVA/UCIT/train/task4.json configs/data_configs/UCIT/IconQA.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task4.json configs/data_configs/UCIT/IconQA.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 4
# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task5.json configs/data_configs/UCIT/CLEVR-Math.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task5.sh configs/model_configs/LLaVA/UCIT/train/task5.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task5.json configs/data_configs/UCIT/CLEVR-Math.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 5
# pip install -e .
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_pre/task6.json configs/data_configs/UCIT/Flickr30k.json 0.2 fixed_rank
bash scripts/Train_UCIT/Task6.sh configs/model_configs/LLaVA/UCIT/train/task6.json configs/data_configs/UCIT/Flickr30k.json
bash scripts/Train_UCIT/extract_gradients.sh configs/model_configs/LLaVA/UCIT/train_post/task6.json configs/data_configs/UCIT/Flickr30k.json 0.2 energy
bash scripts/Eval_UCIT/Eval_finetune1.sh 6

bash scripts/Eval_UCIT/Eval_finetune1a.sh
