#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:50:38 2017

@author: melgazar9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score




#################################################################
#                                                               #
#                   READ AND CLEAN UP DATA                      #                       
#                                                               #
#################################################################

df_train = pd.read_csv('/Users/melgazar9/Downloads/titanic_train.csv')
df_test = pd.read_csv('/Users/melgazar9/Downloads/titanic_test.csv')

# Reorder df_train to make Survived the last column
df_train = df_train[['PassengerId', 'Pclass', 'Name', 'Sex',
                    'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                    'Embarked', 'Survived']]

# Assign X (data) and y (target)
X = df_train.drop('Survived', axis=1)
y = df_train.Survived


# Fill nan values
X.fillna(X.Age.median(), inplace=True)
df_test.fillna(X.Age.median(), inplace=True)



# Numerical columns are columns that have only numerical values in the cells
#numerical_cols = list(df_train.dtypes[df_train.dtypes != 'object'].index)
#numerical_cols.pop(-1)
#non_numerical_cols = [set(X.columns) - set(numerical_cols)]



# One hot encode the Sex and Embarked column in X
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked'], drop_first=True)

# One hot encode the Cabin column - convert the non-numerical values to numerical values 
X['Cabin'] = pd.Series(pd.Categorical(X.Cabin).codes)
df_test['Cabin'] = pd.Series(pd.Categorical(df_test.Cabin).codes)

# One hot encode the rest of the columns: Name and Ticket
X['Name'] = pd.Series(pd.Categorical(X.Name).codes)
df_test['Name'] = pd.Series(pd.Categorical(df_test.Name).codes)

X['Ticket'] = pd.Series(pd.Categorical(X.Ticket).codes)
df_test['Ticket'] = pd.Series(pd.Categorical(df_test.Ticket).codes)


#################################################################
#                                                               #
#                   MACHINE LEARNING                            #                       
#                                                               #
#################################################################

# Make both data frames have the same features using intersection
#possible_features = [val for val in X_train.columns if val in X_test.columns]
possible_features = [val for val in X.columns if val in df_test.columns]
X = X[possible_features]

# Train test split on the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# Normalization and scaling 
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)




#################################################################
#                                                               #
#                   K NEAREST NEIGHBORS (KNN)                   #                       
#                                                               #
#################################################################


# 10-fold cross-validation with KNN classification algorithm
# Find the best n_neighbors parameter

knn = KNeighborsClassifier()
#scores = cross_val_score(knn, X_train, y_train, scoring='accuracy')
k_range = range(1,31)

#k_scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#    k_scores.append(scores.mean())
#print 'max KNN k-fold CV score: ', max(k_scores)

# Plot it in matplotlib
#plt.bar(k_range, k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross Validation Accuracy')

#StandardScaler


# Or just do it this way using Grid Search
# Measure performance based on accuracy, precision, and recall
param_grid = dict(n_neighbors = k_range)

knn_grid_accuracy = GridSearchCV(knn, param_grid, scoring='accuracy', cv=10).fit(X_train_scaled, y_train)

knn_grid_precision = GridSearchCV(knn, param_grid, scoring='precision', cv=10).fit(X_train_scaled, y_train)

knn_grid_recall = GridSearchCV(knn, param_grid, scoring='recall', cv=10).fit(X_train_scaled, y_train)

knn_grid_f1 = GridSearchCV(knn, param_grid, scoring='f1', cv=10).fit(X_train_scaled, y_train)

knn_grid_auc = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=10).fit(X_train_scaled, y_train)





#knn_grid_accuracy_params = [knn_grid.grid_scores_[i].parameters for i in range(30)]
#knn_grid_accuracy_scores = [knn_grid.grid_scores_[i].cv_validation_scores for i in range(30)]
#knn_grid_accuracy_mean_val_scores = [knn_grid.grid_scores_[i].mean_validation_score for i in range(30)]


knn_grid_accuracy_mean_scores = [result.mean_validation_score for result in knn_grid_accuracy.grid_scores_]
knn_grid_precision_mean_scores = [result.mean_validation_score for result in knn_grid_precision.grid_scores_]
knn_grid_recall_mean_scores = [result.mean_validation_score for result in knn_grid_recall.grid_scores_]
knn_grid_f1_mean_scores = [result.mean_validation_score for result in knn_grid_f1.grid_scores_]
knn_grid_auc_mean_scores = [result.mean_validation_score for result in knn_grid_auc.grid_scores_]

print 'grid accuracy mean scores \n', knn_grid_accuracy_mean_scores
#print 'grid precision mean scores \n', knn_grid_precision_mean_scores
#print 'grid recall mean scores \n', knn_grid_recall_mean_scores

#######################################################################

# Show the results in matplotlib

plt.bar(k_range, knn_grid_accuracy_mean_scores)
plt.title('KNN Accuracy Grid Mean Scores')
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross Validation Accuracy')
plt.show()


plt.bar(k_range, knn_grid_precision_mean_scores)
plt.title('KNN Precision Grid Mean Scores')
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross Validation Precision')
plt.show()


plt.bar(k_range, knn_grid_recall_mean_scores)
plt.title('KNN Recall Grid Mean Scores')
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross Validation Recall')
plt.show()


plt.bar(k_range, knn_grid_f1_mean_scores)
plt.title('KNN F1 Score Grid Mean Scores')
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross Validation F1 Score')
plt.show()


plt.bar(k_range, knn_grid_auc_mean_scores)
plt.title('KNN Grid Mean AUC Scores')
plt.xlabel('Value of K of KNN')
plt.ylabel('Cross Validation AUC Score')
plt.show()


#######################################################################


# Fill knn parameters with grid_search best_params_ for best accuracy
knn_best_accuracy = knn.set_params(**knn_grid_accuracy.best_params_).fit(X_train_scaled, y_train)

# Predict outcomes in target for knn with best accuracy
knn_best_accuracy_y_pred = knn_best_accuracy.predict(X_train_scaled)
print 'KNN best accuracy GridSearchCV Predictions: ', knn_best_accuracy_y_pred
print 'KNN best accuracy value_counts() \n', pd.Series(knn_best_accuracy_y_pred).value_counts()
print 'KNN average cross validation score for best accuracy: ', cross_val_score(knn_best_accuracy, 
                                                                    X_train_scaled, y_train, cv=10, scoring='accuracy').mean()
knn_conf_matrix_best_accuracy = confusion_matrix(y_train, knn_best_accuracy.predict(X_train_scaled))
print 'KNN Confusion Matrix Best Accuracy: \n', knn_conf_matrix_best_accuracy
print 'KNN Best Accuracy Accuracy Score: ', accuracy_score(y_train, knn_best_accuracy.predict(X_train_scaled))
print 'KNN Best Accuracy Classification Report: \n', classification_report(y_train, knn_best_accuracy.predict(X_train_scaled))
print '\n\n'



#######################################################################



# Fill knn parameters with grid_search best_params_ for best precision
knn_best_precision = knn.set_params(**knn_grid_precision.best_params_).fit(X_train_scaled, y_train)

# Predict outcomes in target for knn with best precision
knn_best_precision_y_pred = knn_best_precision.predict(X_train_scaled)
print 'KNN best precision GridSearchCV Predictions: ', knn_best_precision_y_pred
print 'KNN best precision value_counts() \n', pd.Series(knn_best_precision_y_pred).value_counts()
print 'KNN average cross validation score for best precision: ', cross_val_score(knn_best_precision, 
                                                                    X_train_scaled, y_train, cv=10, scoring='precision').mean()

knn_conf_matrix_best_precision = confusion_matrix(y_train, knn_best_precision.predict(X_train_scaled))
print 'KNN Confusion Matrix Best Precision: \n', knn_conf_matrix_best_precision
print 'KNN Best Precision Accuracy Score: ', accuracy_score(y_train, knn_best_precision.predict(X_train_scaled))
print 'KNN Best Precision Classification Report: \n', classification_report(y_train, knn_best_precision.predict(X_train_scaled))
print '\n\n'



#######################################################################



# Fill knn parameters with grid_search best_params_ for best recall
knn_best_recall = knn.set_params(**knn_grid_recall.best_params_).fit(X_train_scaled,y_train)

# Predict outcomes in target for knn with best accuracy
knn_best_recall_y_pred = knn_best_recall.predict(X_train_scaled)
print 'KNN best recall GridSearchCV Predictions: ', knn_best_recall_y_pred
print 'KNN best recall value_counts() \n', pd.Series(knn_best_recall_y_pred).value_counts()
print 'KNN average cross validation score for best recall: \n', cross_val_score(knn_best_recall, 
                                                                    X_train_scaled, y_train, cv=10, scoring='recall').mean()

knn_conf_matrix_best_recall = confusion_matrix(y_train, knn_best_recall.predict(X_train_scaled))
print 'KNN Confusion Matrix Best Recall: \n', knn_conf_matrix_best_recall
print 'KNN Best Recall Accuracy Score: ', accuracy_score(y_train, knn_best_recall.predict(X_train_scaled))
print 'KNN Best Recall Classification Report: \n', classification_report(y_train, knn_best_recall.predict(X_train_scaled))
print '\n\n'


#######################################################################


# Predict outcomes in target for knn with best f1 score
knn_best_f1 = knn.set_params(**knn_grid_f1.best_params_).fit(X_train_scaled,y_train)

# Predict outcomes in target for knn with best accuracy
knn_best_f1_y_pred = knn_best_f1.predict(X_train_scaled)
print 'KNN best F1 GridSearchCV Predictions: ', knn_best_f1_y_pred
print 'KNN best F1 value_counts() \n', pd.Series(knn_best_f1_y_pred).value_counts()
print 'KNN average cross validation score for best F1: \n', cross_val_score(knn_best_f1, 
                                                                    X_train_scaled, y_train, cv=10, scoring='f1').mean()

knn_conf_matrix_best_f1 = confusion_matrix(y_train, knn_best_f1.predict(X_train_scaled))
print 'KNN Confusion Matrix Best F1: \n', knn_conf_matrix_best_f1
print 'KNN Best F1 Accuracy Score: ', accuracy_score(y_train, knn_best_f1.predict(X_train_scaled))
print 'KNN Best F1 Classification Report: \n', classification_report(y_train, knn_best_f1.predict(X_train_scaled))
print '\n\n'


#######################################################################


# Fill knn parameters with grid_search best_params_ for best AUC score
knn_best_auc = knn.set_params(**knn_grid_auc.best_params_).fit(X_train_scaled,y_train)

# Predict outcomes in target for knn with best auc
knn_best_auc_y_pred = knn_best_auc.predict(X_train_scaled)
print 'KNN best AUC GridSearchCV Predictions: ', knn_best_auc_y_pred
print 'KNN best AUC value_counts() \n', pd.Series(knn_best_auc_y_pred).value_counts()
print 'KNN average cross validation score for best AUC: ', cross_val_score(knn_best_auc, 
                                                                    X_train_scaled, y_train, cv=10, scoring='roc_auc').mean()
knn_conf_matrix_best_auc = confusion_matrix(y_train, knn_best_auc.predict(X_train))
print 'KNN Confusion Matrix Best AUC: \n', knn_conf_matrix_best_auc

print 'KNN Best AUC Accuracy Score: ', accuracy_score(y_train, knn_best_accuracy.predict(X_train_scaled))
print 'KNN Best AUC Classification Report: \n', classification_report(y_train, knn_best_auc.predict(X_train_scaled))
print '\n\n'


#######################################################################


# Number of neighbors to get best accuracy, precision, and recall
print 'KNN best accuracy will happen with ', knn_grid_accuracy.best_params_, ' neighbors.'
print 'KNN best precision will happen with ', knn_grid_precision.best_params_, ' neighbors.'
print 'KNN best recall will happen with ', knn_grid_recall.best_params_, ' neighbors.'
print 'KNN best F1 score will happen with ', knn_grid_f1.best_params_, ' neighbors.'
print 'KNN best AUC score will happen with ', knn_grid_auc.best_params_, ' neighbors.'





knn_unseen_predictions = knn_grid_accuracy.predict(df_test)







#answer = pd.DataFrame(y_predict_rf_test_set).set_index(df_test.PassengerId)
#answer.to_csv('/Users/melgazar9/Desktop/TitanicSolutions.csv')