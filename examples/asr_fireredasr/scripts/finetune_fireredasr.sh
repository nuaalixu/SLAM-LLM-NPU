#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1



run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/asr_fireredasr
train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/train/
dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/dev/
firered_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/FireRedASR-LLM/
output_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU/examples/aispeech_asr/exp-$(date +"%Y%m%d")
# ckpt_path=
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_path=$llm_path \
++model_config.firered_path=$firered_path \
++model_config.encoder_path=$speech_encoder_path \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.warmup_steps=1000 \
++train_config.total_steps=110000 \
++train_config.lr=5e-5 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

    # --num_nodes 1 \
    # --num_gpus 8 \


torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --master_port=29505 \
    $code_dir/finetune_fireredasr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=true \
    ${hydra_args}
