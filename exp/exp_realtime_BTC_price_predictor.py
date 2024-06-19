import torch
import time
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
import os
import warnings
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class RealTimeBTCPricePredictor(Exp_Basic):
    def __init__(self, args):
        super(RealTimeBTCPricePredictor, self).__init__(args)
        self.times = []
        self.pred_prices = []
        self.true_prices = []

        # Set up Plotly plot
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.pred_price_trace = go.Scatter(x=[], y=[], mode='lines', name='Predicted Price')
        self.true_price_trace = go.Scatter(x=[], y=[], mode='lines', name='True Price')
        self.fig.add_trace(self.pred_price_trace, secondary_y=True)
        self.fig.add_trace(self.true_price_trace, secondary_y=True)
        
        self.fig.update_layout(
            title="Real-Time BTC Price Prediction",
            xaxis_title="Time",
            yaxis_title="Price",
        )
        
        self.fig.update_xaxes(rangeslider_visible=True)
        self.fig.show()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def update_plot(self, time_last, pred_price, true_price):
        self.times.append(time_last)
        self.pred_prices.append(pred_price)
        self.true_prices.append(true_price)

        if len(self.times) > 96:
            self.times.pop(0)
            self.pred_prices.pop(0)
            self.true_prices.pop(0)

        # Debugging prints
        print(f"Updated times: {self.times}")
        print(f"Predicted prices: {self.pred_prices}")
        print(f"True prices: {self.true_prices}")

        self.fig.data[0].x = self.times
        self.fig.data[0].y = self.pred_prices
        self.fig.data[1].x = self.times
        self.fig.data[1].y = self.true_prices
        self.fig.update_layout(
            title="Real-Time BTC Price Prediction",
            xaxis_title="Time",
            yaxis_title="Price",
        )
        self.fig.show()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
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
        
        try:
            while True:
                start_time = time.time()
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

                        pred = outputs
                        true = batch_x

                        pred_first = pred[0, :, -1][0]
                        true_last = true[0, :, -1][-1]
                        time_last = test_data.get_last_time()

                        print(f"Prediction at {time.strftime('%Y-%m-%d %H:%M:%S')} : {pred[0, :, -1][0]}")
                        print(f"True at {time.strftime('%Y-%m-%d %H:%M:%S')} : {true[0, :, -1][-1]}")
                        print('get_last_time', test_data.get_last_time())
                        self.update_plot(time_last, pred_first, true_last)

                while time.time() - start_time < 60:
                    time.sleep(1)

        except KeyboardInterrupt:
            print("Real-time prediction stopped by user.")
        finally:
            pass