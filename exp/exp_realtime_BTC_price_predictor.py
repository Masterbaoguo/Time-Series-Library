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
        # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/rtBTC', 'checkpoint.pth')))
        # self.model.eval()

        # 加载检查点
        checkpoint_path = os.path.join('./checkpoints/rtBTC', 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_path)

        # 获取当前模型的 state_dict
        model_dict = self.model.state_dict()

        # 过滤不匹配的参数
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        ignored_keys = [k for k in checkpoint.keys() if k not in model_dict]

        # 打印未加载的键
        if ignored_keys:
            print("Warning: The following keys in the checkpoint were not loaded into the model:")
            for key in ignored_keys:
                print(f"  {key}")

        # 更新模型权重
        model_dict.update(filtered_checkpoint)
        self.model.load_state_dict(model_dict)

        # 打印提示信息
        print(f"Model loaded from {checkpoint_path} with {len(filtered_checkpoint)} keys successfully loaded, {len(ignored_keys)} keys ignored.")
        

        while True:  # 进入实时预测的循环
            start_time = time.time()
            # 重新获取数据加载器
            test_data, test_loader = self._get_data(flag='test')

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        
                        batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(shape)
                        # batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    # batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_x

                    # 打印或记录预测结果
                    print(f"Prediction at {time.strftime('%Y-%m-%d %H:%M:%S')} : {pred[0, :, -1][0]}")
                    print(f"true at {time.strftime('%Y-%m-%d %H:%M:%S')} : {true[0, :, -1][-1]}")
                    print('get_last_time', test_data.get_last_time())
                    # print(f"true at {time.strftime('%Y-%m-%d %H:%M:%S')} : {true[0, :, 0]}")

            # 等待下一分钟
            time_to_wait = 60 - (time.time() - start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
