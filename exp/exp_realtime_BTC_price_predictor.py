import torch
import matplotlib.pyplot as plt
import time
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
import os
import warnings

warnings.filterwarnings('ignore')
import time
import numpy as np
import matplotlib.pyplot as plt

class RealTimeBTCPricePredictor(Exp_Basic):
    def __init__(self, args):
        super(RealTimeBTCPricePredictor, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 传递 num_workers=0 确保 DataLoader 在主线程上运行
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/rtBTC/checkpoint.pth')))

        # self.model.eval()
 
        # 初始化实时绘图
        plt.ion()  # 打开交互模式
        fig, ax = plt.subplots()
        true_line, = ax.plot([], [], label='True Price')
        pred_line, = ax.plot([], [], label='Predicted Price')
        ax.legend()

        while True:  # 进入实时预测的循环
            start_time = time.time()
            # 更新数据，假设是你持续或定期更新数据的操作
            test_data.update_data()  # 更新数据集
            # 重新获取数据加载器
            test_data, test_loader = self._get_data(flag='test')

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    print('i', i)
                    print('batch_x', batch_x)
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
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
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    # 打印或记录预测结果
                    print(f"Prediction at {time.strftime('%Y-%m-%d %H:%M:%S')} : {pred}")

                    # 更新实时绘图
                    true_line.set_data(np.arange(len(true[0, :, 0])), true[0, :, 0])
                    pred_line.set_data(np.arange(len(pred[0, :, 0])) + len(true[0, :, 0]), pred[0, :, 0])
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

            # 等待下一分钟
            time_to_wait = 60 - (time.time() - start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        plt.ioff()  # 关闭交互模式
        plt.show()