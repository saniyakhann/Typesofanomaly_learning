import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

#load the data
data = pd.read_csv("data.csv")
print(data.head())

#filter data and set index
data = data.loc[data["logical_site_id"] == 6420]
data["period_start_time"] = pd.to_datetime(data["period_start_time"])
data = data.set_index("period_start_time")
data = data["rrc_accessibility"]
print(data.head())

#prepare data for Isolation Forest
data_df = data.reset_index()
data_values = data_df[["rrc_accessibility"]].values

#fit Isolation Forest
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
data_df['anomaly'] = isolation_forest.fit_predict(data_values)

#identify anomalies
anomalies = data_df[data_df['anomaly'] == -1]

#plot the results
fig = px.line(data_df, x='period_start_time', y='rrc_accessibility', title='RRC Accessibility Over Time')

#add anomalies
fig.add_scatter(x=anomalies['period_start_time'], y=anomalies['rrc_accessibility'],
                mode='markers', marker=dict(color='red', size=8), name='Anomalies')

fig.show()
