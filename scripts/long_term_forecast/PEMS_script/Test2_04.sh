export CUDA_VISIBLE_DEVICES=1

model_name=Test2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/zzyy/Time_main/dataset/PEMS/PEMS04/ \
  --data_path pems04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --patience 5 \
  --use_norm 0 \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/zzyy/Time_main/dataset/PEMS/PEMS04/ \
  --data_path pems04.npz \
  --model_id PEMS04_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --patience 5 \
  --use_norm 0 \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/zzyy/Time_main/dataset/PEMS/PEMS04/ \
  --data_path pems04.npz \
  --model_id PEMS04_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --patience 5 \
  --use_norm 0 \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/zzyy/Time_main/dataset/PEMS/PEMS04/ \
  --data_path pems04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --patience 5 \
  --use_norm 0 \
  --itr 1 \
