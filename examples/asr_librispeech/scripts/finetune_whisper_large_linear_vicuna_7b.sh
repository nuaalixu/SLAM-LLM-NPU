#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export PATH=$PATH:~/../lihaoyu/tools/miniconda3/bin      # ffmpeg

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

speech_encoder_path=~/../lihaoyu/models/Whisper/large-v3.pt
llm_path=~/../lihaoyu/models/vicuna-7b-v1.5
train_data_path=~/../lihaoyu/datasets/LibriSpeech/formats/slam-llm_asr/dev-clean.json #/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl
val_data_path=~/../lihaoyu/datasets/LibriSpeech/formats/slam-llm_asr/dev-clean.json #/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl

output_dir=exp/whisper-large-v3_linear_vicuna-7b-v1.5 #$(date +"%Y%m%d")
npus=1

. ~/../lihaoyu/tools/scripts/parse_options.sh

export ASCEND_VISIBLE_DEVICES=$(seq -s ',' 0 $((npus - 1)))

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/train.log \
++dataset_config.file=slam_llm/datasets/speech_dataset.py:get_speech_dataset \
++model_config.file=slam_llm/models/slam_model.py:model_factory \
"

# -m debugpy --listen 5678 --wait-for-client
# if [[ $ASCEND_VISIBLE_DEVICES != *","* ]]; then
if [[ ${npus} == 1 ]]; then
    # python -m debugpy --listen 5678 --wait-for-client finetune_asr.py \
    python finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
