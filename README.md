# Typesofanomaly_learning
Comparative analysis of supervised and unsupervised machine learning approaches for anomaly detection, evaluating performance across different algorithms and datasets. 

Supervised learning: 
Training a model based on a labeled dataset; training example is labelled with an output label 
The model is trained to minimise the error between its predictions and actual labels 

Applications: 
classification -> eg: spam detection 
regression -> eg: predicting home prices 

- using isolationforest -> from the sk learn library
- using adtk detector -> this also helps us identify different TYPES of anomalies including: QuantileAD, thresholdAD, levelshiftAD, seasonalAD

Unsupervised learning: 
Training a model based on an unlabeled dataset where a model tries to find patterns within the data with explicit instructions 
The model learning to group data point based on inherent structures in the data 
This is useful when you don't have labelled examples of anomalies. 

- Finds data in file
- splits data into training and testing sets. 
