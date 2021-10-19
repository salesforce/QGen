gpu=$1
model=$2
dataset=$3
output_dir=$4
bs=$5

CUDA_VISIBLE_DEVICES=${gpu} python run_qg.py \
--model_name_or_path ${model} \
--dataset_dir ${dataset} \
--output_dir ${output_dir} \
--do_eval \
--predict_with_generate True \
--per_device_eval_batch_size=${bs} \
--run_name ${output_dir} \
--report_to none \
--max_target_length 32 \
--val_max_target_length 32 \
--metric_for_best_model eval_rougeLsum \
