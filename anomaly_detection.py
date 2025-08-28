import pandas as pd
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, QuantileAD

#load the data
data = pd.read_csv("data-20240706.csv")
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
plot(data, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
plt.show()
