#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1



run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/aispeech_asr

dataset=aishell-1
task=hotword
train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/train
dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/dev
train_max_frame_length=1600
eval_max_frame_length=1000
multitask_prompt_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multiprompt.jsonl"
# prompt_style="\{\}\\<speech\\>" # "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" | "USER: {}\n ASSISTANT:"
projector=linear
llm_name=Qwen2.5-7B-Instruct
use_peft=true # For llm
use_fp16=true
freeze_encoder=false
# use absolute path
deepspeed_config=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU/examples/aispeech_asr/conf/ds_config.json

if [[ $use_peft == "true"  ]];then
    ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU/examples/aispeech_asr/exp/20250502-0908-aishell-1-lorafalse/aispeech_asr_epoch_27_step_100
fi

# Choose Encoder
encoder_name=conformer
speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/conformer
encoder_dim=768

# Choose LLM
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-1.5B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-1.5B-Instruct
    llm_dim=3584 
else
    exit 1
fi






output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++dataset_config.train_max_frame_length=$train_max_frame_length \
++dataset_config.eval_max_frame_length=$eval_max_frame_length \
++dataset_config.multitask_prompt_path=$multitask_prompt_path \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.model_name=aispeech_asr \
++train_config.num_epochs=50 \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.freeze_llm=true \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=80 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++metric=acc \
"
if [[ $use_peft == "true"  ]];then
    hydra_args+="++ckpt_path=$ckpt_path/pytorch_model.bin"
fi

# deepspeed \
#     --num_nodes 1 \
#     --num_gpus 8 \
#     $code_dir/finetune_deepspeed.py \
#     --config-path "conf" \
#     --config-name "prompt.yaml" \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=$use_fp16 \
#     ++deepspeed_config=$deepspeed_config \
#     ${hydra_args}


HOST_FILE="/tmp/"${JobID}                        #生成的hostfile的完整文件名，$JobID调度系统会自动生成
 
echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}

deepspeed \
    --node_rank=$RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --hostfile $HOST_FILE \
    --no_ssh \
    $code_dir/finetune_deepspeed.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}
