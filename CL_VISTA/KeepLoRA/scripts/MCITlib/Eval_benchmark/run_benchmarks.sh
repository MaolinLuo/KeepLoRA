HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Eval_benchmark/run_qa_longvideobench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/General_benchmark/longvideobench.json $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_mmbench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/General_benchmark/mmbench-video.json $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_nextqa.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/General_benchmark/nextqa.json $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_qa_mmvu.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/General_benchmark/mmvu.json $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
bash scripts/MCITlib/Eval_benchmark/run_benchmark_mvbench.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/General_benchmark/mvbench.json $HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/task8.json
