import torch
import torch.nn as nn
import os
import warnings
from datetime import timedelta
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.display_manager import DisplayManager
from utils.BTC_trading_strategy import BTCTradingStrategy

warnings.filterwarnings('ignore')

class RealTimeBTCPricePredictor(Exp_Basic):
    def __init__(self, args):
        super(RealTimeBTCPricePredictor, self).__init__(args)
        self.pred_time = 5
        self.interval = 1
        self.trade_interval = 1
        self.time = 0
        self.display = DisplayManager(self, self.pred_time, self.interval)
        self.trade = BTCTradingStrategy()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def predict(self):
        self.time += self.interval
        test_data, test_loader = self._get_data(flag='test')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                pred_prices = outputs[0, :, -1][:self.pred_time]  # next 10 minutes predictions
                true_last = batch_x[0, :, -1][-1]
                time_last = test_data.get_last_time()

                print(f"Predictions for next {self.pred_time} minutes starting from {time_last + timedelta(minutes=1)} : {pred_prices}")
                print(f"True at {time_last} : {true_last}")
                self.display.update_data(time_last, true_last, pred_prices)
                if(self.time % self.trade_interval == 0):
                    self.trade.update(true_last, pred_prices[self.trade_interval - 1])


    def test(self, setting, test=0):
        checkpoint_path = os.path.join('./checkpoints/rtBTC', 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_path)
        model_dict = self.model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        ignored_keys = [k for k in checkpoint.keys() if k not in model_dict]

        if ignored_keys:
            print("Warning: The following keys in the checkpoint were not loaded into the model:")
            for key in ignored_keys:
                print(f"  {key}")

        model_dict.update(filtered_checkpoint)
        self.model.load_state_dict(model_dict)
        print(f"Model loaded from {checkpoint_path} with {len(filtered_checkpoint)} keys successfully loaded, {len(ignored_keys)} keys ignored.")
        self.display.run()