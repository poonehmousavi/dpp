#!/bin/bash

# Define dataset values and corresponding checkpoints
declare -A CHECKPOINTS=(
  # ["libriasr"]="libriasr_residual_400_160/202502172132/checkpoint_29.pth"
  # ["librisqa"]="librisqa_residual_400_160/202502172132/checkpoint_29.pth"
  # ["er"]="er_residual_400_160/202502172132/checkpoint_29.pth"
  # ["clotho_audio_cap"]="clotho_audio_cap_residual_400_160/202502172132/checkpoint_29.pth"
  # ["cv_trans"]="cv_trans_residual_400_160/202502172132/checkpoint_29.pth"
  ["voxceleb_sv"]="voxceleb_sv_residual_400_160/202502180435/checkpoint_21.pth"
)

EVAL_DIR="evaluations/single_task/residual/"

for dataset in "${!CHECKPOINTS[@]}"; do
  EXP_NAME="tmp"
  CHECKPOINT=${CHECKPOINTS[$dataset]}

  echo "Running evaluation for dataset: $dataset (Experiment: $EXP_NAME, Checkpoint: $CHECKPOINT)"

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
            model.soft_prompts=False \
            model.l2p=True \
	    model.prompt_strategy=residual_strategy \
	    model.pool_size=400 \
	    model.prompt_size=160 \
	    model.stochastic=False \
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

