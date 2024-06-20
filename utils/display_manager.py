import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
from datetime import timedelta

class DisplayManager:
    def __init__(self, predictor, pred_time):
        self.predictor = predictor
        self.times_true = []
        self.true_prices = []
        self.pred_data = []  # List to hold future predictions
        self.pred_time = pred_time

    def update_data(self, time_last, true_last, pred_prices):
        self.times_true.append(time_last)
        self.true_prices.append(float("{:.2f}".format(true_last)))  # current true price, rounded to 2 decimal places

        if len(self.times_true) > 96:
            self.times_true.pop(0)
            self.true_prices.pop(0)

        pred_prices = [float("{:.2f}".format(p)) for p in pred_prices.tolist()]

        self.pred_data.append({
            'start_time': time_last,
            'times': [time_last + timedelta(minutes=i) for i in range(self.pred_time + 1)],  # ensure the first time matches the true data time
            'prices': [true_last] + pred_prices  # include the true last price as the first prediction
        })

        if len(self.pred_data) > 96:
            self.pred_data.pop(0)

    def get_figure(self):
        fig = go.Figure()

        true_price_trace = go.Scatter(
            x=self.times_true, 
            y=self.true_prices, 
            mode='lines+markers', 
            name='True Price', 
            marker=dict(size=6),
            line=dict(width=4)  # 加粗线宽
        )
        fig.add_trace(true_price_trace)

        for i, pred in enumerate(self.pred_data):
            color = f'rgba({(i*37)%256},{(i*97)%256},{(i*157)%256},0.8)'
            pred_trace = go.Scatter(
                x=pred['times'], 
                y=pred['prices'], 
                mode='lines+markers', 
                name=f'Predicted Price {i+1}', 
                line=dict(color=color, dash='dash'), 
                marker=dict(size=6)
            )
            fig.add_trace(pred_trace)

        fig.update_layout(
            title="Real-Time BTC Price Prediction",
            xaxis_title="Time",
            yaxis_title="Price",
        )

        fig.update_yaxes(tickformat=".2f")

        return fig
    
    def run(self):
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Real-Time BTC Price Prediction"),
            dcc.Graph(id="live-graph", style={'height': '100vh', 'width': '100vw'}),
            dcc.Interval(id="update-interval", interval=60*1000, n_intervals=0)  # 每分钟刷新一次
        ])

        @app.callback(
            Output("live-graph", "figure"),
            Input("update-interval", "n_intervals")
        )
        def update_graph(n_intervals):
            self.predictor.predict()
            return self.get_figure()

        app.run_server(debug=True, use_reloader=False)