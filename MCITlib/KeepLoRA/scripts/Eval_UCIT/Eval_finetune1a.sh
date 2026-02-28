#!/bin/bash

# Task 1 evaluated on future tasks' datasets
bash scripts/Eval_UCIT/eval_arxivqa.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/ArxivQA.json
bash scripts/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/VizWiz.json
bash scripts/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/IconQA.json
bash scripts/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task1.json configs/data_configs/UCIT/Flickr30k.json

# Task 2 evaluated on future tasks' datasets
bash scripts/Eval_UCIT/eval_vizwiz.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/VizWiz.json
bash scripts/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/IconQA.json
bash scripts/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task2.json configs/data_configs/UCIT/Flickr30k.json

# Task 3 evaluated on future tasks' datasets
bash scripts/Eval_UCIT/eval_iconqa.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/IconQA.json
bash scripts/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task3.json configs/data_configs/UCIT/Flickr30k.json

# Task 4 evaluated on future tasks' datasets
bash scripts/Eval_UCIT/eval_clevr.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/CLEVR-Math.json
bash scripts/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task4.json configs/data_configs/UCIT/Flickr30k.json

# Task 5 evaluated on future tasks' datasets
bash scripts/Eval_UCIT/eval_flickr30k.sh configs/model_configs/LLaVA/UCIT/eval/task5.json configs/data_configs/UCIT/Flickr30k.json

# Task 6 has no future tasks (last task)
# No evaluation needed for upper triangular