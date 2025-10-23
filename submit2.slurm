#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --job-name=musubi-train
#SBATCH --output=logs/JN-%j.log
#SBATCH --account=bffz-delta-gpu

# module load gcc
# module load python/miniforge3_pytorch/2.7.0

source /u/ykwon4/.bashrc
conda activate /u/ykwon4/myenv

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
    --dit /home/yo564250/workspace/ComfyUI/models/diffusion_models/FramePackI2V_HY_bf16.safetensors \
    --vae /home/yo564250/workspace/ComfyUI/models/vae/hunyuan-video-t2v-720p-vae.pt \
    --text_encoder1 /home/yo564250/workspace/ComfyUI/models/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 /home/yo564250/workspace/ComfyUI/models/text_encoders/clip_l.safetensors \
    --image_encoder /home/yo564250/workspace/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors \
    --dataset_config /groups/chenchen/patrick/opens2v_chara2.toml \
    --sdpa --mixed_precision bf16 --one_frame \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 8 --persistent_data_loader_workers --split_attn \
    --network_weights /home/yo564250/workspace/whisperer/related/framepackbase/musubi-tuner/outputs/training/idmask_control_lora_wrope_v2/idmask_control_lora_wrope_v2_5-step00006000.safetensors \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 9 --save_every_n_steps 1000 --seed 42 \
    --sample_prompts /groups/chenchen/patrick/opens2v_chara2.txt \
    --sample_every_n_steps 250 --sample_at_first \
    --output_dir outputs/training/idmask_control_lora_wrope_v2_multi --output_name idmask_control_lora_wrope_v2_multi_tmp \
    --logging_dir outputs/training/idmask_control_lora_wrope_v2_multi/logs --log_with tensorboard \
    --remove_embedding --use_attention_controlimage_masking --sample_with_latentbbox_rope
date