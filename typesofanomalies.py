import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from adtk.data import validate_series
from adtk.detector import QuantileAD, ThresholdAD, LevelShiftAD, SeasonalAD

#load the data
data = pd.read_csv("data.csv")
print(data.head())

#filter data and set index
data = data.loc[data["logical_site_id"] == 6420]
data["period_start_time"] = pd.to_datetime(data["period_start_time"])
data = data.set_index("period_start_time")
data = data["rrc_accessibility"]
print(data.head())

#validate series
data = validate_series(data)

#ensure regular frequency
data = data.asfreq('H')  # Assuming hourly frequency, adjust as needed

#handle NaN values
data = data.interpolate(method='time')  # Interpolate missing values

#re-validate series
data = validate_series(data)

#initialize detectors
quantile_detector = QuantileAD(low=0.01, high=0.99)
threshold_detector = ThresholdAD(high=99.95, low=99.05)
level_shift_detector = LevelShiftAD(c=6.0, side='both', window=10)
seasonal_detector = SeasonalAD(freq=168)  # 7 days * 24 hours = 168 hours for weekly seasonality

#detect anomalies
quantile_anomalies = quantile_detector.fit_detect(data)
threshold_anomalies = threshold_detector.detect(data)
level_shift_anomalies = level_shift_detector.fit_detect(data)
seasonal_anomalies = seasonal_detector.fit_detect(data)

#create subplots
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=('QuantileAD Anomalies', 'ThresholdAD Anomalies', 'LevelShiftAD Anomalies', 'SeasonalAD Anomalies'))

#plot data and anomalies
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Data'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Data'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Data'), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Data'), row=4, col=1)

#add anomaly points
fig.add_trace(go.Scatter(x=data.index[quantile_anomalies == 1], y=data[quantile_anomalies == 1],
                         mode='markers', marker=dict(color='red', size=8), name='Quantile Anomalies'), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index[threshold_anomalies == 1], y=data[threshold_anomalies == 1],
                         mode='markers', marker=dict(color='blue', size=8), name='Threshold Anomalies'), row=2, col=1)

fig.add_trace(go.Scatter(x=data.index[level_shift_anomalies == 1], y=data[level_shift_anomalies == 1],
                         mode='markers', marker=dict(color='green', size=8), name='Level Shift Anomalies'), row=3, col=1)

fig.add_trace(go.Scatter(x=data.index[seasonal_anomalies == 1], y=data[seasonal_anomalies == 1],
                         mode='markers', marker=dict(color='purple', size=8), name='Seasonal Anomalies'), row=4, col=1)

#update layout
fig.update_layout(height=1200, width=1000, title='Different Types of Anomalies Detected')
fig.show()



