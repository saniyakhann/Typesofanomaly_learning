import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#load the wine dataset containing chemical analysis of three wine types
wine = load_wine()
#extract feature data (chemical measurements)
X = wine.data  
#extract target labels (wine types: 0, 1, or 2)
Y = wine.target  

#split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#initialise standard scaler to normalise the feature data
sc = StandardScaler()
#fit the scaler to training data and transform it
X_train = sc.fit_transform(X_train)
#transform test data using the same scaling parameters
X_test = sc.transform(X_test)

#create support vector classifier with radial basis function kernel
svmclassifier = SVC(C=1.0, kernel='rbf', random_state=0)
#train the classifier on the scaled training data
svmclassifier.fit(X_train, Y_train)

#make predictions on the test set
Y_pred = svmclassifier.predict(X_test)

#calculate the accuracy of the model on the test set
accuracy = svmclassifier.score(X_test, Y_test)

#create confusion matrix to evaluate classification performance
cm = confusion_matrix(Y_test, Y_pred)

#calculate precision (type 1 error metric) - weighted average for multi-class
type1error = precision_score(Y_test, Y_pred, average='weighted')

#calculate recall (type 2 error metric) - weighted average for multi-class
type2error = recall_score(Y_test, Y_pred, average='weighted')

#print evaluation metrics for wine classification
print("----prediction----")
print("")
print("accuracy: %.2f%%" % (accuracy * 100.0))
print("precision: %.2f%%" % (type1error * 100.0))
print("recall: %.2f%%" % (type2error * 100.0))
print("")

#create dataframe for predictions with appropriate column naming
YP_pred_df = pd.DataFrame(Y_pred, columns=['Y^'])
#set index name for the dataframe
YP_pred_df.index.name = 'Index'

#save predictions to a csv file for future reference
YP_pred_df.to_csv('winesvm_predictions.csv', index=True)
