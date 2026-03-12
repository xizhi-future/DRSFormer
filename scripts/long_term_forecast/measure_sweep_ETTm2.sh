export CUDA_VISIBLE_DEVICES=1
python measure_sweep.py \
  --mode both \
  --topk_list "2,6,12,21" \
  --center_list "1,2,4,8" \
  --batch_size 8 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --device cuda:0 \
  --use_real_data 1 \
  --real_batch_count 1 \
  --root_path /home/zzyy/Time_main/dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --dataset ETTm2 \
  --output_dir ./measure_results_opt/ETTm2_real
