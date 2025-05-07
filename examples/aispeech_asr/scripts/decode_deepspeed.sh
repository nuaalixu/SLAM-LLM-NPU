#!/bin/bash
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/aispeech_asr

projector=linear
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=true
use_fp16=false
eval_max_frame_length=2000
dataset=aishell-1
task=asr
ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU/examples/aispeech_asr/exp/20250506-0955-aishell-1-loratrue_hotword_instruct/aispeech_asr_epoch_2_step_560

test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/test


# Choose Encoder
encoder_name=conformer
speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/conformer
encoder_dim=768

# Choose LLM
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/vicuna-7b-v1.5
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-7B
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-7B" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B
    llm_dim=3584 
else
    exit 1
fi

decode_log=$ckpt_path/decode_$dataset_${task}
deepspeed \
    --num_nodes 1 \
    --num_gpus 8 \
    $code_dir/inference_batch_deepspeed.py \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.normalize=true \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=aispeech_asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=dynamic \
    ++train_config.num_epochs=1 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/pytorch_model.bin



python examples/aispeech_asr/scripts/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer
python examples/aispeech_asr/scripts/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer
