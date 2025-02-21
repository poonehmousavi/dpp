#!/bin/bash

# Define dataset values and corresponding checkpoints
declare -A CHECKPOINTS=(
   # ["er"]="er_sp_160_stochastic/202502180429/checkpoint_29.pth"
   ["libriasr"]="libriasr_sp_160_stochastic/202502180147/checkpoint_27.pth"
   # ["librisqa"]="librisqa_sp_160_stochastic/202502180425/checkpoint_29.pth"
   # ["voxceleb_sv"]="voxceleb_sv_sp_160_stochastic/202502180429/checkpoint_29.pth"
   # ["clotho_audio_cap"]="clotho_audio_cap_sp_160_stochastic/202502180153/checkpoint_29.pth"
   # ["cv_trans"]="cv_trans_sp_160_stochastic/202502180230/checkpoint_29.pth"
)

EVAL_DIR="evaluations/single_task/sp_stochastic/"

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
            model.soft_prompts=True \
            model.l2p=False \
	    model.num_soft_prompt_tokens=160 \
	    model.stochastic=True \
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

