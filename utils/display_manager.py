import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
from datetime import timedelta

class DisplayManager:
    def __init__(self, predictor, pred_time, interval, trading_strategy):
        self.predictor = predictor
        self.times_true = []
        self.true_prices = []
        self.pred_data = []  # List to hold future predictions
        self.pred_time = pred_time
        self.interval = interval
        self.trading_strategy = trading_strategy  # BTCTradingStrategy instance
        self.asset_times = []
        self.asset_values = []

    def update_data(self, time_last, true_prices, pred_prices):
        self.times_true.append(time_last)
        true_last = true_prices[-1]
        self.true_prices.append(float("{:.2f}".format(true_last)))  # current true price, rounded to 2 decimal places

        # Update trading strategy and record asset value
        self.trading_strategy.update(true_prices, pred_prices)
        self.asset_times.append(time_last)
        self.asset_values.append(self.trading_strategy.asset_history[-1])

        if len(self.times_true) > 96:
            self.times_true.pop(0)
            self.true_prices.pop(0)
            self.asset_times.pop(0)
            self.asset_values.pop(0)

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

        # BTC True Price trace
        true_price_trace = go.Scatter(
            x=self.times_true, 
            y=self.true_prices, 
            mode='lines+markers', 
            name='True Price', 
            marker=dict(size=6),
            line=dict(width=4),  # 加粗线宽
            yaxis='y1'
        )
        fig.add_trace(true_price_trace)

        # BTC Predicted Prices traces
        for i, pred in enumerate(self.pred_data):
            color = f'rgba({(i*37)%256},{(i*97)%256},{(i*157)%256},0.8)'
            pred_trace = go.Scatter(
                x=pred['times'], 
                y=pred['prices'], 
                mode='lines+markers', 
                name=f'Predicted Price {i+1}', 
                line=dict(color=color, dash='dash'), 
                marker=dict(size=6),
                yaxis='y1'
            )
            fig.add_trace(pred_trace)

        # Asset Value trace
        asset_value_trace = go.Scatter(
            x=self.asset_times,
            y=self.asset_values,
            mode='lines+markers',
            name='Total Asset Value',
            marker=dict(size=6, color='red'),
            line=dict(width=4, color='red'),  # 设置线宽和颜色
            yaxis='y2'
        )
        fig.add_trace(asset_value_trace)

        # Update layout to add secondary y-axis
        fig.update_layout(
            title="Real-Time BTC Price Prediction and Asset Value",
            xaxis_title="Time",
            yaxis=dict(
                title="BTC Price",
                tickformat=".2f",
                titlefont=dict(color='#1f77b4'),
                tickfont=dict(color='#1f77b4')
            ),
            yaxis2=dict(
                title="Total Asset Value",
                overlaying='y',
                side='right',
                tickformat=".2f",
                titlefont=dict(color='red'),
                tickfont=dict(color='red')
            )
        )

        return fig
    
    def run(self):
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Real-Time BTC Price Prediction and Asset Value"),
            dcc.Graph(id="live-graph", style={'height': '100vh', 'width': '95vw'}),
            dcc.Interval(id="update-interval", interval=self.interval*60*1000, n_intervals=0)  # 每self.interval分钟刷新一次
        ])

        @app.callback(
            Output("live-graph", "figure"),
            Input("update-interval", "n_intervals")
        )
        def update_graph(n_intervals):
            self.predictor.predict()
            return self.get_figure()

        app.run_server(debug=True, use_reloader=False)
