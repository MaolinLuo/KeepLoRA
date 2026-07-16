#!/bin/bash

set -e

bash scripts/Train_DCL/extract_weights.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task0.json 0.6

# pip install -e .
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task1.json configs/data_configs/MLLM-DCL/RS.json 0.2 fixed_rank
bash scripts/Train_DCL/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task1.json configs/data_configs/MLLM-DCL/RS.json
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_post/task1.json configs/data_configs/MLLM-DCL/RS.json 0.2 energy
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 1
# pip install -e .
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task2.json configs/data_configs/MLLM-DCL/Med.json 0.2 fixed_rank
bash scripts/Train_DCL/Task2.sh configs/model_configs/LLaVA/MLLM-DCL/train/task2.json configs/data_configs/MLLM-DCL/Med.json
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_post/task2.json configs/data_configs/MLLM-DCL/Med.json 0.2 energy
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 2
# pip install -e .
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task3.json configs/data_configs/MLLM-DCL/AD.json 0.2 fixed_rank
bash scripts/Train_DCL/Task3.sh configs/model_configs/LLaVA/MLLM-DCL/train/task3.json configs/data_configs/MLLM-DCL/AD.json
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_post/task3.json configs/data_configs/MLLM-DCL/AD.json 0.2 energy
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 3
# pip install -e .
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task4.json configs/data_configs/MLLM-DCL/Sci.json 0.2 fixed_rank
bash scripts/Train_DCL/Task4.sh configs/model_configs/LLaVA/MLLM-DCL/train/task4.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_post/task4.json configs/data_configs/MLLM-DCL/Sci.json 0.2 energy
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 4
# pip install -e .
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_pre/task5.json configs/data_configs/MLLM-DCL/Fin.json 0.2 fixed_rank
bash scripts/Train_DCL/Task5.sh configs/model_configs/LLaVA/MLLM-DCL/train/task5.json configs/data_configs/MLLM-DCL/Fin.json
bash scripts/Train_DCL/extract_gradients.sh configs/model_configs/LLaVA/MLLM-DCL/train_post/task5.json configs/data_configs/MLLM-DCL/Fin.json 0.2 energy
bash scripts/Eval_MLLM_DCL/Eval_finetune1.sh 5

bash scripts/Eval_MLLM_DCL/Eval_finetune1a.sh