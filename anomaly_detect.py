import pandas as pd
import plotly.express as px
from adtk.data import validate_series
from adtk.detector import QuantileAD

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

#quantile-based anomaly detection
quantile_detector = QuantileAD(low=0.01, high=0.99)
anomalies = quantile_detector.fit_detect(data)

#plot the results
fig = px.line(data.reset_index(), x='period_start_time', y='rrc_accessibility', title='RRC Accessibility Over Time')

#add anomalies
anomaly_points = data[anomalies == 1].reset_index()
fig.add_scatter(x=anomaly_points['period_start_time'], y=anomaly_points['rrc_accessibility'],
                mode='markers', marker=dict(color='red', size=8), name='Anomalies')

fig.show()
