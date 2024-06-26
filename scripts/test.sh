#训练：
#11个参数 20240626
python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path BTC_USDT-2020-01-01-1m.csv  --model_id BTCh_96_96  --model PatchTST  --data BTC  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 11  --dec_in 11  --c_out 11  --des 'Exp'  --n_heads 2  --batch_size 32  --itr 1 --freq t --inverse
#20个参数
python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path cleaned_BTC_USDT-2020-01-01-1m.csv  --model_id BTCh_96_96  --model PatchTST  --data BTC  --features MS  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 20  --dec_in 20  --c_out 20  --des 'Exp'  --n_heads 8  --batch_size 128  --itr 1 --freq t --inverse
#20个参数DLine
python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/btc/  --data_path BTC_USDT-2024-01-01-1m.csv  --model_id BTCh_96_96  --model DLinear  --data BTC  --features MS  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 20  --dec_in 20  --c_out 20  --des 'Exp'  --n_heads 8  --batch_size 128  --itr 1 --freq t

#测试
cd D:\project\Time-Series-Library\Time-Series-Library
conda activate TTime-Series-Library

python -u run.py  --task_name btc_price_forecast  --is_training 0 --model_id BTCh_96_96  --model PatchTST  --data rtBTC  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 11  --dec_in 11  --c_out 11  --des 'Exp'  --n_heads 2  --batch_size 1  --itr 1 --num_workers 0  --freq t --inverse

python -u run.py  --task_name btc_price_forecast  --is_training 0 --model_id BTCh_96_96  --model PatchTST  --data rtBTC  --features M  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 11  --dec_in 11  --c_out 11  --des 'Exp'  --n_heads 2  --batch_size 1  --itr 1 --num_workers 0  --freq t --inverse --backtest

python -u run.py  --task_name btc_price_forecast  --is_training 0 --model_id BTCh_96_96  --model DLinear  --data rtBTC  --features MS  --seq_len 96  --label_len 48  --pred_len 96  --e_layers 1  --d_layers 1  --factor 3  --enc_in 20  --dec_in 20  --c_out 20  --des 'Exp'  --n_heads 8  --batch_size 1  --itr 1 --num_workers 0  --freq t --inverse --backtest