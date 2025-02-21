#!/bin/bash

# Define dataset values
DATASETS=("voxceleb_sv" "libriasr" "librisqa" "er" "clotho_audio_cap" "cv_trans")
EVAL_DIR="evaluations/multi_task/sp_stochastic_1/"
CHECKPOINT="multitask_sp_160_stochastic/202502180158/checkpoint_29.pth"

for dataset in "${DATASETS[@]}"; do
  EXP_NAME="tmp"
  echo "Running evaluation for dataset: $dataset (Experiment: $EXP_NAME)"
  

  # Launch the job with the current configuration
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
    --env EXP_NAME="$EXP_NAME" \
    -- bash -c "
        source /home/toolkit/.bashrc;
        source activate salmon;
        cd /home/toolkit/SALMONN/;
        python3 train.py \
            --cfg-path configs/config_llama.yaml \
            --options run.run_name=$EXP_NAME \
	    model.lora=False \
            model.soft_prompts=True \
            model.l2p=False \
            model.prompt_strategy=sim_strategy \
            model.pool_size=400 \
            model.prompt_size=1 \
            model.stochastic=True \
	    model.num_soft_prompt_tokens=160 \
            run.eval_dir=$EVAL_DIR \
            run.output_dir=$EXP_NAME \
            datasets.dataset=$dataset \
            run.eval_split=test \
            model.ckpt=$CHECKPOINT \
            run.batch_size_eval=1 \
            run.evaluate=True \
            run.num_valid_iters=-1;
    " &

  sleep 2  # Small delay to avoid overloading the job scheduler
done

wait  # Wait for all jobs to finish before exiting the script

