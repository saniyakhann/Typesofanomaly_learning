import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score


X, Y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=0)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


svmclassifier = SVC(C=1.0, kernel='rbf', random_state=0)
svmclassifier.fit(X_train, Y_train)


Y_pred = svmclassifier.predict(X_test)


accuracy = svmclassifier.score(X_test, Y_test)


cm = confusion_matrix(Y_test, Y_pred)


type1error = precision_score(Y_test, Y_pred, average='weighted')


type2error = recall_score(Y_test, Y_pred, average='weighted')

print("----Prediction----")
print("")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (type1error * 100.0))
print("Recall: %.2f%%" % (type2error * 100.0))
print("")


YP_pred_df = pd.DataFrame(Y_pred, columns=['Y^'])
YP_pred_df.index.name = 'Index'


YP_pred_df.to_csv('RandomSVM_Predictions.csv', index=True)
