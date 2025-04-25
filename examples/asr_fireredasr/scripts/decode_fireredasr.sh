#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
set -e 
run_dir=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/SLAM-LLM-NPU/
cd $run_dir
code_dir=examples/asr_fireredasr
npu_device=7 # choose npu to decode on
dataset=librispeech-other
dataset_task=asr
prompt_style=normal  
use_peft=true # fireredasr has llm-lora
use_fp16=true
pad_or_trim=true
ckpt_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/FireRedASR/pretrained_models/FireRedASR-LLM-L-finetuned/v3.0/FireRedASR-LLM-L # ckpt for fireredasr
file=examples/asr_fireredasr/model/slam_fireredasr.py:model_factory

# to do list: place your own test dataset
if [[ $dataset == "aishell-1" || $dataset == "aishell-2" || $dataset == "alimeeting" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/mandarin/${dataset}/${dataset_task}/test/
elif [[ $dataset == "librispeech-other" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-other/
elif [[ $dataset == "librispeech-clean" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-clean/
elif [[ $dataset == "cantonese" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/test/King-ASR-300/export_llm
elif [[ $dataset == "ws-net" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/mandarin/wenetspeech/asr/test_net
elif [[ $dataset == "ws-meeting" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/mandarin/wenetspeech/asr/test_meeting
elif [[ $dataset == "sichuan" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/test/sichuan_300/export_llm
elif [[ $dataset == "jiangsu" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/test/aify_suzhou_datatang_apr20v1/export_llm
elif [[ $dataset == "shanghai" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/test/aify_shanghai_mdtest_dec20v1/export_llm
elif [[ $dataset == "aispeech_meeting" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/test/aispeech_meeting/export_llm
else
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/test/
fi

get_dataset=examples/asr_fireredasr/model/slam_fireredasr.py:get_speech_testdataset
decode_log=examples/asr_fireredasr/decode_log/decode_${dataset}_${dataset_task}_${prompt_style}
# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_fireredasr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$ckpt_path \
    ++model_config.file=$file \
    ++model_config.ckpt_path=$ckpt_path \
    ++model_config.normalize=true \
    ++dataset_config.prompt_style=$prompt_style \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.pad_or_trim=$pad_or_trim \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++dataset_config.cmvn_file=$ckpt_path/cmvn.ark \
    ++dataset_config.file=$get_dataset \
    ++train_config.model_name=firered_asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=10 \
    ++train_config.num_workers_dataloader=8\
    ++train_config.output_dir=$output_dir \
    ++train_config.inference_mode=true \
    ++train_config.npu_device=$npu_device \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/model.pth.tar

# you can design and place your wer computation file here:
if [[ $dataset == "librispeech-clean" || $dataset == "librispeech-other" ]]
then  
    ref=${decode_log}_gt
    out=${decode_log}_pred
    python /aistor/aispeech/hpc_stor01/home/pengjing00sx/tools/wer/wer.py --print_sentence_wer 1 --do_tn 0 --rm_special 0 --ref $ref --hyp $out > $out.wer 2>&1
    tail -n8 $out.wer
else
    python /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/tools/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer 
fi
