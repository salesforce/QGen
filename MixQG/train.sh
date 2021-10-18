num_gpus=$1 # 4
model_name=$2 # t5-base
dataset=$3 # squad
output_dir=$4
lr=$5 # 3e-5
bs=$6 # 32

deepspeed --num_gpus=${num_gpus} run_qg.py \
--model_name_or_path ${model_name} \
--dataset_dir ${dataset} \
--output_dir ${output_dir} \
--do_train \
--do_eval \
--evaluation_strategy steps \
--eval_steps 2000 \
--save_steps 2000 \
--load_best_model_at_end True \
--metric_for_best_model eval_rougeLsum \
--greater_is_better True \
--predict_with_generate True \
--per_device_eval_batch_size=${bs} \
--per_device_train_batch_size=${bs} \
--gradient_accumulation_steps=1 \
--max_steps 100000 \
--logging_steps 100 \
--save_total_limit 4 \
--deepspeed configs/ds_config_zero2.json \
--adam_eps 1e-06 \
--label_smoothing 0.1 \
--learning_rate ${lr} \
--logging_first_step \
--warmup_steps 500 \
--max_target_length 32 \
--val_max_target_length 32 \
--fp16
