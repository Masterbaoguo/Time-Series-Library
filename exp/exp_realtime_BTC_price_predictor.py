import torch
import matplotlib.pyplot as plt
import time

class RealTimeBTCPricePredictor:
    def __init__(self, model, checkpoint_path, fetch_data_func, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.fetch_data_func = fetch_data_func
        
        # Load the trained model
        self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        
    def predict_next(self, latest_data):
        with torch.no_grad():
            batch_x, batch_x_mark = self.prepare_input_data(latest_data)
            
            # Prepare decoder input
            dec_inp = torch.zeros((1, self.pred_len, batch_x.shape[-1])).float().to(self.device)
            dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp], dim=1).float().to(self.device)
            
            if self.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_x_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_x_mark)
                
            outputs = outputs[:, -self.pred_len:, :]
            pred = outputs.detach().cpu().numpy()
            
            if self.inverse:
                pred = self.data.inverse_transform(pred)
            
            return pred[0,:, -1]

    def prepare_input_data(self, latest_data):
        batch_x = latest_data['batch_x'].float().to(self.device).unsqueeze(0)
        batch_x_mark = latest_data['batch_x_mark'].float().to(self.device).unsqueeze(0)
        return batch_x, batch_x_mark
        
    def plot_prediction(self, real_values, predicted_values):
        plt.figure(figsize=(10,5))
        plt.plot(real_values, label='Real Prices')
        plt.plot(predicted_values, label='Predicted Prices')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('BTC Price')
        plt.title('Real-time BTC Price Prediction')
        plt.draw()
        plt.pause(1)  # Pause to update the plot. Adjust as necessary.
        plt.clf()
        
    def run_realtime_prediction(self, interval=60):
        """
        Continuously fetch new data, predict the next price, and plot the results.
        :param interval: Time interval (in seconds) between each fetch-predict-plot cycle.
        """
        real_values = []
        predicted_values = []
        
        while True:
            latest_data = self.fetch_data_func()
            prediction = self.predict_next(latest_data)
            
            real_values.append(latest_data['latest_price'])
            predicted_values.append(prediction)
            
            self.plot_prediction(real_values, predicted_values)
            print(f"Real: {latest_data['latest_price']}, Predicted: {prediction}")
            
            time.sleep(interval)