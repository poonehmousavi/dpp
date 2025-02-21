#!/bin/bash

export EXP_NAME="tmp"

# EAI Job Command
eai job new \
    --restartable \
    --image registry.toolkit-sp.yul201.service-now.com/snow.shared/interactive-toolkit \
    --data snow.research.dssk.data:/mnt/dssk/data:ro \
    --data snow.research.dssk.data:/mnt/dssk/data_rw \
    --data snow.research.dssk.results:/mnt/dssk/results \
    --data snow.research.dssk.shubham_gupta1:/home/toolkit \
    --workdir /home/toolkit \
    --env HOME=/home/toolkit \
    --cpu 32 \
    --mem 320 \
    --gpu 1 \
    --gpu-mem 80 \
    --gpu-model-filter=H100 \
    --env EXP_NAME=$EXP_NAME \
    -- bash -c "
        source /home/toolkit/.bashrc;
        source activate salmon;
        cd /home/toolkit/SALMONN/;
	python3 train.py \
		--cfg-path configs/config_llama.yaml \
		--options run.run_name=$EXP_NAME \
		model.lora=True \
		model.soft_prompts=False \
		model.l2p=False \
		model.num_soft_prompt_tokens=100 \
		model.pool_size=50 \
		model.prompt_size=20 \
		run.output_dir=$EXP_NAME;
    "


