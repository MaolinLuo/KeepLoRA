#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$MODEL_CONFIG" gpu_num)
STAGE=$(read_config "$MODEL_CONFIG" stage)
MODELPATH=$(read_config "$MODEL_CONFIG" model_path)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
ANNOTATION=$(read_config "$DATA_CONFIG" anno_path)
IMAGE=$(read_config "$DATA_CONFIG" image_folder)

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="./results/UCIT/each_dataset/VizWiz"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_others \
        --model-path $MODELPATH \
        --question-file $DATA_PATH \
        --image-folder $IMAGE \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Truncate long text entries
MAX_TEXT_LENGTH=300
temp_file="${output_file}.tmp"
python3 -c "
import json
import sys

max_length = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if line.strip():
            data = json.loads(line)
            if 'text' in data and len(data['text']) > max_length:
                data['text'] = data['text'][:max_length]
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
" "$MAX_TEXT_LENGTH" "$output_file" "$temp_file"

mv "$temp_file" "$output_file"

python -m llava.eval.CoIN.eval_caption \
    --annotation-file $ANNOTATION \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
