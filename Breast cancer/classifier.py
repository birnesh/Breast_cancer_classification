import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.svm import SVC 

df = pd.read_csv("/home/birnesh/Documents/Breast cancer/Breast_Cancer_dataset.csv",index_col=0).dropna(axis=1)
print(df.head())
#print(df.columns )
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
#y =pd.get_dummies(y,drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train,y_train)
clf = SVC(C=1.0, kernel="linear")
nv = GaussianNB()
nvc = nv.fit(X_test, y_test)
clf.fit(X_train, y_train)
#print(classification_report())
y_pred = clf.predict(X_test)
knn_y_pred = model.predict(X_test)
nvc_y_pred = nvc.predict(X_test)
print("SVC: ",classification_report(y_test.values.ravel(), y_pred))
print("SVC: ",accuracy_score(y_test,y_pred))
print("knn :", classification_report(y_test.values.ravel(), knn_y_pred))
print("knn: ",accuracy_score(y_test,knn_y_pred))
print("nvc :", classification_report(y_test.values.ravel(), nvc_y_pred))
print("nvc: ",accuracy_score(y_test,nvc_y_pred))
#print(y_test)
#x_sns = pd.concat([X_test,y_test],axis=1)
#sns.set()
#sns.heatmap(df.isnull())
#sns.pairplot(x_sns,hue="diagnosis")



