export CUDA_VISIBLE_DEVICES=1

python -u measure_sweep.py \
  --mode centers \
  --center_list "2,4,8,16,32" \
  --batch_size 8 \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --device cuda:0 \
  --output_dir ./measure_results/centers
