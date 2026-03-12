export CUDA_VISIBLE_DEVICES=1
python measure_sweep.py \
  --mode both \
  --topk_list "2,6,12,21" \
  --center_list "2,4,8,16" \
  --batch_size 4 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --d_model 512 \
  --device cuda:0 \
  --use_real_data 1 \
  --real_batch_count 1 \
  --root_path /home/zzyy/Time_main/dataset/weather/ \
  --data_path weather.csv \
  --dataset custom \
  --output_dir ./measure_results_opt/Weather_real
