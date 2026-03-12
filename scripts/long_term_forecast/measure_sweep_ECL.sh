export CUDA_VISIBLE_DEVICES=1
python measure_sweep.py \
  --mode both \
  --topk_list "8,32,64,128,321" \
  --center_list "4, 16, 32, 64" \
  --batch_size 16 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --d_model 512 \
  --device cuda:0 \
  --use_real_data 1 \
  --real_batch_count 1 \
  --root_path /home/zzyy/Time_main/dataset/electricity/ \
  --data_path electricity.csv \
  --dataset custom \
  --output_dir ./measure_results_opt/ECL_real_bs1_16
