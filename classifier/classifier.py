#Imports


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn import linear_model
#from sklearn import tree
#from sklearn import neighbors
#from sklearn import ensemble
#from sklearn import svm
#from sklearn import gaussian_process
#from sklearn import naive_bayes
#from sklearn import neural_network
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV

#Reading data

testset = pd.read_csv("/home/birnesh/Documents/classifier/test.csv")
trainset = pd.read_csv("/home/birnesh/Documents/classifier/train.csv")


#plotting to see the varience of the data

sns.set()
sns.pairplot(trainset[["bone_length","rotting_flesh","hair_length","has_soul","color","type"]],hue="type")

#creating new features

trainset['hair_soul'] = trainset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)
trainset['hair_bone'] = trainset.apply(lambda row: row['hair_length']*row['bone_length'],axis=1)
trainset['bone_soul'] = trainset.apply(lambda row: row['bone_length']*row['has_soul'],axis=1)
trainset['hair_soul_bone'] = trainset.apply(lambda row: row['hair_length']*row['has_soul']*row['bone_length'],axis=1)

testset['hair_soul'] = testset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)
testset['hair_bone'] = testset.apply(lambda row: row['hair_length']*row['bone_length'],axis=1)
testset['bone_soul'] = testset.apply(lambda row: row['bone_length']*row['has_soul'],axis=1)
testset['hair_soul_bone'] = testset.apply(lambda row: row['hair_length']*row['has_soul']*row['bone_length'],axis=1)

#creating test and train data
#x = pd.concat([trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul", "hair_bone", "bone_soul", "hair_soul_bone"]], pd.get_dummies(trainset["color"])], axis=1)
x = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul", "hair_bone", "bone_soul", "hair_soul_bone"]]
x_original = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]
x_hair_soul = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]
y = trainset[["type"]]
#x_test = pd.concat([testset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul", "hair_bone", "bone_soul", "hair_soul_bone"]], pd.get_dummies(testset["color"])], axis=1)
x_test = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul", "hair_bone", "bone_soul", "hair_soul_bone"]]
x_test_original = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]
x_test_hair_soul = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]

#creating a dictionary to hold classifier objects
clfs = {}
#clfs['lr'] = {'clf': linear_model.LogisticRegression(), 'name':'LogisticRegression'}
#clfs['rf'] = {'clf': ensemble.RandomForestClassifier(n_estimators=750, n_jobs=-1), 'name':'RandomForest'}
#clfs['knn'] = {'clf': neighbors.KNeighborsClassifier(n_neighbors=4), 'name':'kNearestNeighbors'}
#clfs['svc'] = {'clf': svm.SVC(kernel='linear'), 'name': 'SupportVectorClassifier'}

#some of the classifiers
clfs['tr'] = {'clf': DecisionTreeClassifier(), 'name':'DecisionTree'}
clfs['nusvc'] = {'clf': NuSVC(gamma='scale'), 'name': 'NuSVC'}
clfs['linearsvc'] = {'clf': LinearSVC(), 'name': 'LinearSVC'}
clfs['SGD'] = {'clf': SGDClassifier(max_iter=1000 ,tol=1e-3), 'name': 'SGDClassifier'}
clfs['GPC'] = {'clf': GaussianProcessClassifier(), 'name': 'GaussianProcess'}
clfs['nb'] = {'clf': GaussianNB(), 'name':'GaussianNaiveBayes'}
clfs['bag'] = {'clf': BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5), 'name': "BaggingClassifier"}
clfs['gbc'] = {'clf': GradientBoostingClassifier(), 'name': 'GradientBoostingClassifier'}
#clfs['mlp'] = {'clf': neural_network.MLPClassifier(hidden_layer_sizes=(100,100,100), alpha=1e-5, solver='lbfgs', max_iter=500), 'name': 'MultilayerPerceptron'}

#creating parameters for searching
parameters = {'solver': ['lbfgs'], 'max_iter': [1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12)}
clfs['mlpgrid'] = {'clf': GridSearchCV(MLPClassifier(), parameters,cv=3,iid=True), 'name': 'MLP with GridSearch'}

parameters = {'kernel':['linear', 'sigmoid', 'poly', 'rbf'], 'gamma':np.linspace(0.0,2.0,num=21),'C': np.linspace(0.5,1.5,num=11)}
clfs['svcgrid'] = {'clf': GridSearchCV(SVC(), parameters,cv=3,iid=True), 'name': 'SVC with GridSearch'}

parameters = {'n_estimators':np.arange(64, 1024, step=64)}
clfs['rfgrid'] = {'clf': GridSearchCV(RandomForestClassifier(), parameters,cv=3,iid=True), 'name': 'Random Forest with GridSearch'}

parameters = {'n_neighbors':np.arange(3, 12)}
clfs['knngrid'] = {'clf': GridSearchCV(KNeighborsClassifier(), parameters,cv=3,iid=True), 'name': 'KNN with GridSearch'}

parameters = {'n_estimators':np.arange(3, 12)}
clfs['adagrid'] = {'clf': GridSearchCV(AdaBoostClassifier(), parameters,cv=3,iid=True), 'name': 'AdaBoost with GridSearch'}

parameters = {'C':[1],'tol':[0.0001],'solver': ['newton-cg'], 'multi_class': ['multinomial']}
clfs['lrgrid'] = {'clf': GridSearchCV(LogisticRegression(), parameters,cv=3,iid=True), 'name': 'LogisticRegression with GridSearch'}

#fiting it in csv
#for clf in clfs:
#    clfs[clf]['score'] = cross_val_score(clfs[clf]['clf'], x_original, y.values.ravel(), cv=5)
#    print(clfs[clf]['name'] + ": %0.4f (+/- %0.4f)" % (clfs[clf]['score'].mean(), clfs[clf]['score'].std()*2))
#    #clfs[clf]['score'] = cross_val_score(clfs[clf]['clf'], x, y.values.ravel(), cv=5)
#    #print(clfs[clf]['name'] + " (with all artificial features): %0.4f (+/- %0.4f)" % (clfs[clf]['score'].mean(), clfs[clf]['score'].std()*2))
#    clfs[clf]['score'] = cross_val_score(clfs[clf]['clf'], x_hair_soul, y.values.ravel(), cv=5)
#    print(clfs[clf]['name'] + " (with hair_soul feature): %0.4f (+/- %0.4f)" % (clfs[clf]['score'].mean(), clfs[clf]['score'].std()*2))
#    
#    
#
#    
##using voting classifier    
# classifiers using the hair_soul feature
clfs['vote_hair_soul'] = {'clf': VotingClassifier(estimators=[
            ('svcgrid', clfs['svcgrid']['clf']),
            ('lrgrid', clfs['lrgrid']['clf']),
            ('gbc', clfs['gbc']['clf'])
        ], voting='hard'), 'name': 'VotingClassifierHairSoul'}

# classifiers using the original features
clfs['vote'] = {'clf': VotingClassifier(estimators=[
            ('svcgrid', clfs['svcgrid']['clf']),
            ('lrgrid', clfs['lrgrid']['clf']),
            ('nb', clfs['gbc']['clf'])
        ], voting='hard'), 'name': 'VotingClassifier'}
    
#
#
##fitting the models    
##for clf in clfs:
##    clfs[clf]['clf'].fit(x, y.values.ravel())
#    
clfs['vote_hair_soul']['clf'] = clfs['vote_hair_soul']['clf'].fit(x_hair_soul, y.values.ravel())
#clfs['vote']['clf'] = clfs['vote']['clf'].fit(x_original, y.values.ravel())
    
##doing predictions
##for clf in clfs:
##    clfs[clf]['predictions'] = clfs[clf]['clf'].predict(x_test)
#    
clfs['vote_hair_soul']['predictions'] = clfs['vote_hair_soul']['clf'].predict(x_test_hair_soul)

#print(accuracy_score(clfs['vote_hair_soul']['predictions'], ))
print(len(clfs['vote_hair_soul']['predictions']))
#clfs['vote']['predictions'] = clfs['vote']['clf'].predict(x_test_original)
#
##creating a dataframe for storing the rresults
##for clf in clfs:
##    sub = pd.DataFrame(clfs[clf]['predictions'])
##    pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_" + clfs[clf]['name'] + ".csv", index=False)
#
#sub = pd.DataFrame(clfs['vote_hair_soul']['predictions'])
#pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_" + clfs['vote_hair_soul']['name'] + ".csv", index=False)
#
#sub = pd.DataFrame(clfs['vote']['predictions'])
#pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission_" + clfs['vote']['name'] + ".csv", index=False)