#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
run_dir=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/SLAM-LLM-NPU/
cd $run_dir
code_dir=examples/asr_fireredasr
# multitask 
# dataset=alimeeting
# multitask_asr
dataset=aishell-1
prompt_style=normal #instruct
deepspeed_config=examples/asr_fireredasr/conf/ds_config.json # set config here for deepspeed

if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech || $dataset == alimeeting || $dataset == gigaspeech || $dataset == wenetspeech ]]
then
    dataset_task=asr
fi
use_peft=true
use_fp16=true
freeze_encoder=true
pad_or_trim=true

if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    ckpt_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/FireRedASR-LLM-L # base model ckpt
fi

# to so list: change your own dataset
if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech || $dataset == alimeeting || $dataset == gigaspeech || $dataset == wenetspeech ]]
then
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/dev/
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/test/
elif [[  $dataset == "librispeech" ]]
then
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/dev-other/
else
    train_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/collection/${dataset}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/collection/${dataset}/dev/
fi
file=examples/asr_fireredasr/model/slam_fireredasr.py:model_factory
inference_mode=False
get_dataset=examples/asr_fireredasr/model/slam_fireredasr.py:get_speech_testdataset # you can refer to slam_fireredasr.py since there are multiple types of datasets: iterable or map-style(have 'len()'')
output_dir=${code_dir}/exp/${dataset}/$(date +"%Y%m%d")/${encoder_name}_${projector}_${llm_name}_encoder${freeze_encoder}_lora${use_peft}_pad${pad_or_trim}_${prompt_style}_${dataset_task}_speed${speed_perturb}_specaug${spec_augmentation}-$(date +"%H%M")
hydra_args="
hydra.run.dir=$output_dir \
++model_config.ckpt_path=$ckpt_path \
++model_config.normalize=true \
++model_config.file=$file \
++dataset_config.prompt_style=$prompt_style \
++dataset_config.normalize=true \
++dataset_config.dataset=$dataset \
++dataset_config.input_type=$input_type \
++dataset_config.pad_or_trim=$pad_or_trim \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++dataset_config.file=$get_dataset \
++train_config.model_name=fireredasrllm \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-7 \
++train_config.validation_interval=100 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++train_config.inference_mode=$inference_mode \
++train_config.pth_tar=true \
++metric=acc \
"
if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    hydra_args+="++ckpt_path=$ckpt_path"
fi
# hydra_args+="++ckpt_path=$ckpt_path/model.pt"

# -m debugpy --listen 5678 --wait-for-client
if [[ $ASCEND_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_fireredasr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port=29505 \
        $code_dir/finetune_fireredasr_deepspeed.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=$use_fp16 \
        ++deepspeed_config=$deepspeed_config \
        ${hydra_args}
fi
# you can also use deepspeed to start here, the same with aispeech_asr example