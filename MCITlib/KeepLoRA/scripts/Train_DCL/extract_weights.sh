#!/bin/bash

MODEL_CONFIG=$1
ENERGY_THRESHOLD=$2

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

OUTPUT_DIR=$(read_config "$MODEL_CONFIG" output_dir)

python -m llava.train.extract_weights \
    --model-config $MODEL_CONFIG \
    --output-dir $OUTPUT_DIR \
    --energy-threshold $ENERGY_THRESHOLD
