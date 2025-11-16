#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --job-name=musubi-train
#SBATCH --output=logs/JN-%j.log
#SBATCH --account=bffz-dtai-gh

# module load gcc
# module load python/miniforge3_pytorch/2.7.0
# module load cuda/12.6

source /u/ykwon4/.bashrc
conda activate myenv

echo "-------------------------------"
echo "Job Information:"
date
hostname
pwd
module list
echo "SLURM Nodes:  ${SLURM_NODELIST}"
echo "-------------------------------"

mkdir $HOME/tmp >> /dev/null  2>&1
export XDG_RUNTIME_DIR=$HOME/tmp

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 fpack_train_network.py \
    --dit /projects/bffz/ykwon4/ComfyUI/models/diffusion_models/FramePackI2V_HY_bf16.safetensors \
    --vae /projects/bffz/ykwon4/ComfyUI/models/vae/hunyuan-video-t2v-720p-vae.pt \
    --text_encoder1 /projects/bffz/ykwon4/ComfyUI/models/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 /projects/bffz/ykwon4/ComfyUI/models/text_encoders/clip_l.safetensors \
    --image_encoder /projects/bffz/ykwon4/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors \
    --dataset_config /projects/bffz/ykwon4/test.toml \
    --sdpa --mixed_precision bf16 --one_frame \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 8 --persistent_data_loader_workers --split_attn \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 3 --save_every_n_steps 1000 --seed 42 \
    --sample_prompts /projects/bffz/ykwon4/test_sample_prompts.txt \
    --sample_every_n_steps 500 --sample_at_first \
    --output_dir /work/nvme/bffz/ykwon4/musubi-training/idmask_control_lora_wrope_v3 \
    --output_name idmask_control_lora_wrope_v3-lambda000 \
    --logging_dir /work/nvme/bffz/ykwon4/musubi-training/idmask_control_lora_wrope_v3/logs --log_with all \
    --remove_embedding --use_attention_controlimage_masking --sample_with_latentbbox_rope --lambda_mask 0.0
date