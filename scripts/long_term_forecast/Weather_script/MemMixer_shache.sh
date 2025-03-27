export CUDA_VISIBLE_DEVICES=0

model_name=MemMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_6 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 6 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_12 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 12 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_18 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 18 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_24 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 24 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_30 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 30 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_36 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 36 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_42 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 42 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_final_shache_processed.csv \
  --model_id weather_shache_48 \
  --model $model_name \
  --data custom \
  --features M \
  --target T \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 48 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window 