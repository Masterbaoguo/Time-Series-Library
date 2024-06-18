python -u run.py --task_name short_term_forecast --is_training 1 --root_path ./dataset/m4 --seasonal_patterns 'Monthly' --model_id m4_Monthly --model TimesNet --data m4 --features M --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --batch_size 16 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --learning_rate 0.001 --loss 'SMAPE'


python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_96 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1


python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv  --model_id ETTh1_96_96  --model TimesNet  --data ETTh1  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 2  --d_layers 1  --factor 3  --enc_in 7  --dec_in 7  --c_out 7  --d_model 16  --d_ff 32  --des 'Exp'  --itr 1  --top_k 5 


python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTm1.csv  --model_id ETTm1_96_96  --model TimesNet  --data ETTm1  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 2  --d_layers 1  --factor 3  --enc_in 7  --dec_in 7  --c_out 7  --des 'Exp'  --d_model 64  --d_ff 64  --top_k 5  --itr 1


python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path btc_usdt_1m_data.csv  --model_id BTCm1_96_96  --model TimesNet  --data BTC  --features MS  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 2  --d_layers 1  --factor 3  --enc_in 5  --dec_in 5  --c_out 5  --des 'Exp'  --d_model 64  --d_ff 64  --top_k 5  --itr 1 --target close --freq t

python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path btc-2013-01-01-1h.csv  --model_id BTCh_96_96  --model PatchTST  --data custom  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 11  --dec_in 11  --c_out 11  --des 'Exp'  --n_heads 2  --batch_size 32  --itr 1


python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path btc-2021-01-01.csv  --model_id BTCh_96_96  --model PatchTST  --data BTC  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 5  --dec_in 5  --c_out 5  --des 'Exp'  --n_heads 2  --batch_size 32  --itr 1


cd D:\project\Time-Series-Library\Time-Series-Library
conda activate TTime-Series-Library

python -u run.py  --task_name btc_price_forecast  --is_training 0  --root_path ./dataset/btc/  --data_path btc-2013-01-01-1h.csv  --model_id BTCh_96_96  --model PatchTST  --data rtBTC  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 11  --dec_in 11  --c_out 11  --des 'Exp'  --n_heads 2  --batch_size 1  --itr 1 --num_workers 0
