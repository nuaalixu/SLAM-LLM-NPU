#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PATH=$PATH:~/../lihaoyu/tools/miniconda3/bin      # ffmpeg
# export CUDA_LAUNCH_BLOCKING=1

speech_encoder_path=~/../lihaoyu/models/Whisper/large-v3.pt
llm_path=~/../lihaoyu/models/vicuna-7b-v1.5

val_data_path=~/../lihaoyu/datasets/LibriSpeech/formats/slam-llm_asr/test-clean.json

ckpt_path=
infer_tag=

npus=1

tr_batch=4
cv_batch=4

. ~/../lihaoyu/tools/scripts/parse_options.sh

output_dir=$(dirname ${ckpt_path})/infer/${infer_tag}

hydra_args="
hydra.run.dir=${output_dir} \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=${llm_path} \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=${speech_encoder_path} \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.val_data_path=${val_data_path} \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.inference_mode=true \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.num_epochs=1 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=${output_dir} \
++log_config.log_file=${output_dir}/decode.log \
++decode_log=${output_dir}/decode \
++ckpt_path=${ckpt_path} \
++dataset_config.file=slam_llm/datasets/speech_dataset.py:get_speech_dataset \
++model_config.file=slam_llm/models/slam_model.py:model_factory \
"
# 

# ++peft_ckpt=$ckpt_path \
# ++train_config.use_peft=true \
# ++train_config.peft_config.r=32 \
# ++dataset_config.normalize=true \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64

if [[ ${npus} == 1 ]]; then
    python inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ${hydra_args}
else
    for i in $(seq 0 $((npus - 1))); do
        python inference_asr_batch.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            ${hydra_args} &
    done
    wait
fi