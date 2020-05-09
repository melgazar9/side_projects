#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:26:07 2017

@author: melgazar9
"""
# Titanic Kaggle Competition: Predict how many people live or die based on various features

#This model will use independent variables sex and age to predict the 
#dependent variable survived.


# Imports

# pandas
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('whitegrid')
#%matplotlib inline

# machine learning
#from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict


df_train = pd.read_csv('/Users/melgazar9/Downloads/titanic_train.csv')
df_test = pd.read_csv('/Users/melgazar9/Downloads/titanic_test.csv')

# Reorder df_train to make Survived the last column
df_train = df_train[['PassengerId', 'Pclass', 'Name', 'Sex',
                    'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                    'Embarked', 'Survived']]


X = df_train.drop('Survived', axis=1)
y = df_train.Survived
#print(X.head())
#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Show the columns that df_train and df_test have in common
possible_features = [val for val in df_train.columns if val in df_test.columns]
#print(possible_features)

numerical_variables = list(df_train.dtypes[df_train.dtypes != 'object'].index)
numerical_variables.pop(-1)
#print(numerical_variables)


# Define a dummy model for a benchmark
# predict the majority class every time and see the result
#dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X, y)
#y_predict_dummy = dummy_majority.fit(X_train, y_train)

count_majorities = df_train['Survived'].value_counts() 
# 0 is most frequent, i.e. most people died
y_dummy_survived = pd.Series()
df_train['y_dummy_survived'] = y_dummy_survived 
y_dummy_predict = df_train['y_dummy_survived'].fillna(0)# fill with most frequent class

y_dummy_train_predict = y_dummy_predict[0:668]
y_dummy_test_predict = y_dummy_predict[668:]
# Find classification report for dummy classifier
dummy_accuracy = accuracy_score(y_test, y_dummy_test_predict)
dummy_precision = precision_score(y_test, y_dummy_test_predict)
dummy_recall = recall_score(y_test, y_dummy_test_predict)
dummy_f1 = f1_score(y_test, y_dummy_test_predict)
print 'dummy accuracy: ', dummy_accuracy
print 'dummy precision: ', dummy_precision
print 'dummy recall: ', dummy_recall
print 'dummy f1 score: ', dummy_f1



# numerical variables only
X_train_numeric = X_train[numerical_variables]
X_test_numeric = X_test[numerical_variables]
# Fill in ages with the mean age
X_train_numeric.Age.fillna(X_train_numeric['Age'].median(), inplace=True)
X_test_numeric.Age.fillna(X_train_numeric['Age'].median(), inplace=True)

X_train_numeric.fillna(0, inplace=True)
X_test_numeric.fillna(0, inplace=True)


# Define a logistic regression classifier
logistic_clf = LogisticRegression().fit(X_train_numeric, y_train)
y_predict_logistic = logistic_clf.predict(X_test_numeric)
logistic_accuracy = accuracy_score(y_test, y_predict_logistic)
logistic_precision = precision_score(y_test, y_predict_logistic)
logistic_recall = recall_score(y_test, y_predict_logistic)
logistic_f1 = f1_score(y_test, y_predict_logistic)
logistic_auc = roc_auc_score(y_test, y_predict_logistic)
print 'logistic train numeric accuracy: ', logistic_accuracy
print 'logistic train numeric precision: ', logistic_precision
print 'logistic train numeric recall: ', logistic_recall
print 'logistic train numeric f1 score: ', logistic_f1
print 'logistic train numeric auc score: ', logistic_auc



# Get more features using pd.get_dummies
X_train = pd.concat([X_train_numeric, X_train.Embarked, X_train.Sex, X_train.Cabin],
                    axis=1)
X_test = pd.concat([X_test_numeric, X_test.Embarked, X_test.Sex, X_test.Cabin],
                   axis=1)

X_train = pd.get_dummies(X_train, columns = ['Sex', 'Embarked', 'Cabin'], drop_first=True)
X_test = pd.get_dummies(X_test, columns = ['Sex', 'Embarked', 'Cabin'], drop_first=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)


# Find the intersection of features between train and test sets
for train_feature in X_train.columns:
    if train_feature not in X_test.columns:
        X_train.drop(train_feature, axis=1, inplace=True)
        
for test_feature in X_test.columns:
    if test_feature not in X_train.columns:
        X_test.drop(test_feature, axis=1, inplace=True)


# Logistic Regression on dataset with new features
logistic_clf = LogisticRegression().fit(X_train, y_train)
rf_clf = RandomForestClassifier().fit(X_train, y_train)
knn_clf = KNeighborsClassifier().fit(X_train, y_train)
#rf_reg = RandomForestRegressor().fit(X_train, y_train)

y_predict_logistic = logistic_clf.predict(X_test)
y_predict_rf = rf_clf.predict(X_test)
y_predict_knn = knn_clf.predict(X_test)



y_pred_prob_rf = rf_clf.predict_proba(X_test)




logistic_accuracy = accuracy_score(y_test, y_predict_logistic)
logistic_precision = precision_score(y_test, y_predict_logistic)
logistic_recall = recall_score(y_test, y_predict_logistic)
logistic_f1 = f1_score(y_test, y_predict_logistic)
logistic_auc = roc_auc_score(y_test, y_predict_logistic)
print 'logistic train accuracy: ', logistic_accuracy
print 'logistic train precision: ', logistic_precision
print 'logistic train recall: ', logistic_recall
print 'logistic train f1 score: ', logistic_f1
print 'logistic train auc score: ', logistic_auc




y_predict_proba_logistic = logistic_clf.predict_proba(X_test)
#print y_predict_proba_logistic

y_pred_logistic_using_predict_proba = pd.Series()
i=0
while i < len(y_predict_proba_logistic[:,0]):
    if y_predict_proba_logistic[i,0] >= .39:
        y_pred_logistic_using_predict_proba.set_value(i, 0)
        i+=1
    else:
        y_pred_logistic_using_predict_proba.set_value(i, 1)
        i+=1
        
#print 'Logistic Series Predict Classes using predict Proba > .63: ', y_pred_logistic_using_predict_proba
y_pred_logistic_pred_proba_array = np.array(y_pred_logistic_using_predict_proba)
logistic_pred_proba_accuracy = accuracy_score(y_test, y_pred_logistic_pred_proba_array)
logistic_pred_proba_precision = precision_score(y_test, y_pred_logistic_pred_proba_array)
logistic_pred_proba_recall = recall_score(y_test, y_pred_logistic_pred_proba_array)
logistic_pred_proba_f1 = f1_score(y_test, y_pred_logistic_pred_proba_array)
logistic_pred_proba_auc = roc_auc_score(y_test, y_pred_logistic_using_predict_proba)
print 'logistic train accuracy w/ pred_proba: ', logistic_pred_proba_accuracy
print 'logistic train precision w/ pred_proba: ', logistic_pred_proba_precision
print 'logistic train recall w/ pred_proba: ', logistic_pred_proba_recall
print 'logistic train f1 score w/ pred_proba: ', logistic_pred_proba_f1
print 'logistic train auc score w/ pred_proba: ', logistic_pred_proba_auc





# Feature engineer the test set

# numerical variables only
X_test_set_numeric = df_test[numerical_variables]
# Fill in ages with the mean age
median_age = pd.concat([df_train['Age'], df_test['Age']]).median()
X_test_numeric.Age.fillna(median_age, inplace=True)

X_test_set_numeric = pd.concat([X_test_set_numeric, df_test.Embarked,
                                df_test.Sex, df_test.Cabin], axis=1)

X_test_set_numeric = pd.get_dummies(X_test_set_numeric, 
                                    columns = ['Sex', 'Embarked', 'Cabin'], 
                                    drop_first=True)

X_test_set_numeric.fillna(99, inplace=True)


X_test.fillna(0, inplace=True)


# Find the intersection of features between train and test sets

for test_feature in X_test_set_numeric.columns:
    if test_feature not in X_train.columns:
        X_test_set_numeric.drop(test_feature, axis=1, inplace=True)
        
for train_feature in X_train.columns:
    if train_feature not in X_test_set_numeric.columns:
        X_train.drop(train_feature, axis = 1, inplace=True)
       
        
# Fit different classifiers        
logistic_clf = LogisticRegression().fit(X_train, y_train)
rf_clf = RandomForestClassifier(n_estimators=1000, max_features=7, 
                                min_samples_leaf = 15).fit(X_train, y_train)
knn_clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)


y_pred_test_logistic = logistic_clf.predict(X_test_set_numeric)
y_pred_test_rf = rf_clf.predict(X_test_set_numeric)
y_pred_test_knn = knn_clf.predict(X_test_set_numeric)


rf_accuracy = accuracy_score(y_test, y_predict_rf)
rf_precision = precision_score(y_test, y_predict_rf)
rf_recall = recall_score(y_test, y_predict_rf)
rf_f1 = f1_score(y_test, y_predict_rf)
rf_auc = roc_auc_score(y_test, y_predict_rf)
print 'RF train accuracy: ', rf_accuracy
print 'RF train precision: ', rf_precision
print 'RF train recall: ', rf_recall
print 'RF train f1 score: ', rf_f1
print 'RF train auc score: ', rf_auc


# PLOT ROC CURVE
#fpr, tpr, threshold = roc_curve(y_test, y_predict_rf)
#plt.plot(fpr, tpr)
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.title('ROC Curve Random Forest')
#plt.xlabel('FPR (1-Specificity)')
#plt.ylabel('TPR (Sensitivity)')
#plt.grid(True)








knn_accuracy = accuracy_score(y_test, y_predict_knn)
knn_precision = precision_score(y_test, y_predict_knn)
knn_recall = recall_score(y_test, y_predict_knn)
knn_f1 = f1_score(y_test, y_predict_knn)
knn_auc = roc_auc_score(y_test, y_predict_logistic)
print 'KNN train accuracy: ', knn_accuracy
print 'KNN train precision: ', knn_precision
print 'KNN train recall: ', knn_recall
print 'KNN train f1 score: ', knn_f1
print 'KNN train auc score: ', knn_auc


#sns.pairplot(pd.concat([X_train_numeric, y_train], axis=1), 
#             hue='Survived', diag_kind='hist')

#plt.figure()
#plt.bar(x = X_train_numeric['SibSp'], y = y_train, marker='o')
#plt.show()







# Apply dimensionality reduction

# PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Get a huge training set with many features because of one hot encoding
X_train_encoded = pd.get_dummies(X_train, columns = X_train.columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns = X_test.columns, drop_first=True)
X_test_set_encoded = pd.get_dummies(df_test, columns = X_test.columns, drop_first=True)

#for train_feature in X_train_encoded.columns:
#    if train_feature not in X_test_set_encoded.columns:
#        X_train_encoded.drop(train_feature, axis = 1, inplace=True)
#        
#for test_feature in X_test_set_encoded.columns:
#    if test_feature not in X_train_encoded.columns:
#        X_test_set_encoded.drop(test_feature, axis=1, inplace=True)

X_train_normalized = StandardScaler().fit_transform(X_train_encoded)

pca2d = PCA(n_components = 2).fit(X_train_normalized)
X_pca2d = pca2d.transform(X_train_normalized)

#from adspy_shared_utilities import plot_labelled_scatter
#plot_labelled_scatter(X_pca2d, y_test)
plt.scatter(X_pca2d[:,0],X_pca2d[:,1], c=y_train)

rf_clf = RandomForestClassifier(n_estimators=1000, 
                                min_samples_leaf = 15).fit(X_pca2d, y_train)
rf_pca_score = rf_clf.score(X_pca2d, y_train)
print rf_pca_score
#y_pred_rf = rf_clf.predict(X_test_encoded) # this is the problem error

rf_accuracy_pca = accuracy_score(y_test, y_predict_rf)
rf_precision_pca = precision_score(y_test, y_predict_rf)
rf_recall_pca = recall_score(y_test, y_predict_rf)
rf_f1_pca = f1_score(y_test, y_predict_rf)
rf_auc_pca = roc_auc_score(y_test, y_predict_rf)
print 'RF_pca train accuracy: ', rf_accuracy_pca
print 'RF_pca train precision: ', rf_precision_pca
print 'RF_pca train recall: ', rf_recall_pca
print 'RF_pca train f1 score: ', rf_f1_pca
print 'RF_pca train auc score: ', rf_auc_pca


#X_test_set = PCA(n_components=2).fit_transform(X_test)
#y_pred_rf_test_pca = rf_clf.predict(X_test_set)




# PLOT ROC CURVE
#fpr, tpr, threshold = roc_curve(y_test, y_predict_rf)
#fpr, tpr, threshold = roc_curve(y_test, y_pred_prob_rf) # error
#plt.plot(fpr, tpr)
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.title('Random Forest ROC Curve')
#plt.xlabel('FPR (1-Specificity)')
#plt.ylabel('TPR (Sensitivity)')
#plt.grid(True)



#cross_validation_rf = cross_val_predict(rf_clf, X_train, y_train, cv=5)
#cross_validation_rf.predict_proba()


df_answer = pd.DataFrame(y_pred_test_rf)
df_answer.columns = ['Survived']
df_answer.set_index(df_test['PassengerId'], inplace=True)
print df_answer, len(df_answer)
df_answer.to_csv('/Users/melgazar9/Desktop/TitanicSolution.csv')