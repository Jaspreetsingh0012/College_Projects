# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:19:34 2020

@author: prana
"""
#%%
#loading libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import *
import pycaret as pc

#%%
#loading dataset

data = pd.read_csv("application_data.csv")
data.head()

#%%
#data visualization

print("\n------------Data Summary------------\n")
data.info()

#data.describe()

# print("Object type values:",np.count_nonzero(data.select_dtypes('object').columns))
# print("___________________________________________________________________________________________")
# print(data.select_dtypes('object').columns)
# print("___________________________________________________________________________________________")

# print("Integer type values:",np.count_nonzero(data.select_dtypes('int').columns))
# print("___________________________________________________________________________________________")
# print(data.select_dtypes('int').columns)
# print("___________________________________________________________________________________________")

# print("Float type values:",np.count_nonzero(data.select_dtypes('float').columns))
# print("___________________________________________________________________________________________")
# print(data.select_dtypes('float').columns)
# print("___________________________________________________________________________________________")

#%%
#label encoding

le = LabelEncoder()
data['NAME_CONTRACT_TYPE'] = le.fit_transform(data['NAME_CONTRACT_TYPE'])
data['CODE_GENDER'] = le.fit_transform(data['CODE_GENDER'])
data['FLAG_OWN_CAR'] = le.fit_transform(data['FLAG_OWN_CAR'])
data['FLAG_OWN_REALTY'] = le.fit_transform(data['FLAG_OWN_REALTY'])
data['NAME_TYPE_SUITE'] = le.fit_transform(data['NAME_TYPE_SUITE'].astype(str))
data['NAME_INCOME_TYPE'] = le.fit_transform(data['NAME_INCOME_TYPE'])
data['NAME_EDUCATION_TYPE'] = le.fit_transform(data['NAME_EDUCATION_TYPE'])
data['NAME_FAMILY_STATUS'] = le.fit_transform(data['NAME_FAMILY_STATUS'])
data['NAME_HOUSING_TYPE'] = le.fit_transform(data['NAME_HOUSING_TYPE'])
data['OCCUPATION_TYPE'] = le.fit_transform(data['OCCUPATION_TYPE'].astype(str))
data['WEEKDAY_APPR_PROCESS_START'] = le.fit_transform(data['WEEKDAY_APPR_PROCESS_START'])
data['ORGANIZATION_TYPE'] = le.fit_transform(data['ORGANIZATION_TYPE'])
data['FONDKAPREMONT_MODE'] = le.fit_transform(data['FONDKAPREMONT_MODE'].astype(str))
data['HOUSETYPE_MODE'] = le.fit_transform(data['HOUSETYPE_MODE'].astype(str))
data['WALLSMATERIAL_MODE'] = le.fit_transform(data['WALLSMATERIAL_MODE'].astype(str))
data['EMERGENCYSTATE_MODE'] = le.fit_transform(data['EMERGENCYSTATE_MODE'].astype(str))

#%%
#missing data interpretation

data = data.interpolate(method ='linear', limit_direction ='forward')
data = data.dropna(axis = 1)
 
#%%
#train test split

X = data.drop(['TARGET'],axis = 1)
X= X.iloc[:25000,:]
target = data['TARGET']
target = target.iloc[:25000]
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,target,test_size=0.3,random_state=0)

#%%
#pycaret

print("\n------------Pycaret Comparison------------\n")

clf = setup(data=data.iloc[:25000,:], target='TARGET',html= False)
compare_models()

#%%
# #creating table

MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)
row_index = 1

#%%
#knn

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(X_train,Y_train)
Y_pred_knn = knn.predict(X_test)


print("\n------------K Nearest Neighbours------------\n")
score = metrics.accuracy_score(Y_test, Y_pred_knn)
score = score*100
print("\nAccuracy score is :",score,"%")

print("\n",confusion_matrix(Y_test,Y_pred_knn))

print("\n",metrics.classification_report(Y_test,Y_pred_knn))

MLA_compare.loc[row_index,'Model'] = 'knn' 
MLA_compare.loc[row_index,'Model Name'] = knn.__class__.__name__
MLA_compare.loc[row_index, 'Train Accuracy'] = round(knn.score(X_train, Y_train), 2)
MLA_compare.loc[row_index, 'Test Accuracy'] = round(knn.score(X_test, Y_test), 2)
MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, Y_pred_knn),2)
MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, Y_pred_knn),2)
MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, Y_pred_knn),2)
row_index+=1

#%%
#decision tree

dt = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0) 
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test) 

print("\n------------Decision Tree------------\n")
score = metrics.accuracy_score(Y_test, Y_pred_dt)
score = score*100
print("\nAccuracy score is :",score,"%")

print("\n",confusion_matrix(Y_test, Y_pred_dt))

print("\n",metrics.classification_report(Y_test,Y_pred_dt))
 
MLA_compare.loc[row_index,'Model'] = 'dt' 
MLA_compare.loc[row_index,'Model Name'] = dt.__class__.__name__
MLA_compare.loc[row_index, 'Train Accuracy'] = round(dt.score(X_train, Y_train), 2)
MLA_compare.loc[row_index, 'Test Accuracy'] = round(dt.score(X_test, Y_test), 2)
MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, Y_pred_dt),2)
MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, Y_pred_dt),2)
MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, Y_pred_dt),2)
row_index+=1

#%%
#random forest 

rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,Y_train)
Y_pred_rf=rf.predict(X_test)

print("\n------------Random Forest------------\n")
score = metrics.accuracy_score(Y_test, Y_pred_rf)
score = score*100
print("\nAccuracy score is :",score,"%")

print("\n",confusion_matrix(Y_test, Y_pred_rf))

print("\n",metrics.classification_report(Y_test,Y_pred_rf))

MLA_compare.loc[row_index,'Model'] = 'rf' 
MLA_compare.loc[row_index,'Model Name'] = rf.__class__.__name__
MLA_compare.loc[row_index, 'Train Accuracy'] = round(rf.score(X_train, Y_train), 2)
MLA_compare.loc[row_index, 'Test Accuracy'] = round(rf.score(X_test, Y_test), 2)
MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, Y_pred_rf),2)
MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, Y_pred_rf),2)
MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, Y_pred_rf),2)
row_index+=1

#%%
#naive bayes

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred_nb = gnb.predict(X_test)

print("\n------------Gaussian Naive Bayes------------\n")
score = metrics.accuracy_score(Y_test, Y_pred_nb)
score = score*100
print("\nAccuracy score is :",score,"%")

print("\n",confusion_matrix(Y_test, Y_pred_nb))

print("\n",metrics.classification_report(Y_test,Y_pred_nb))

MLA_compare.loc[row_index,'Model'] = 'gnb' 
MLA_compare.loc[row_index,'Model Name'] = gnb.__class__.__name__
MLA_compare.loc[row_index, 'Train Accuracy'] = round(gnb.score(X_train, Y_train), 2)
MLA_compare.loc[row_index, 'Test Accuracy'] = round(gnb.score(X_test, Y_test), 2)
MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, Y_pred_nb),2)
MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, Y_pred_nb),2)
MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, Y_pred_nb),2)
row_index+=1

#%%
#logistic regression

lgr = LogisticRegression(max_iter = 10000)
lgr.fit(X_train,Y_train)
Y_pred_lgr = lgr.predict(X_test)

print("\n------------Logistic Regression------------\n")
score = metrics.accuracy_score(Y_test, Y_pred_lgr)
score = score*100
print("\nAccuracy score is :",score,"%")

print("\n",confusion_matrix(Y_test, Y_pred_lgr))

print("\n",metrics.classification_report(Y_test,Y_pred_lgr))

MLA_compare.loc[row_index,'Model'] = 'lgr' 
MLA_compare.loc[row_index,'Model Name'] = lgr.__class__.__name__
MLA_compare.loc[row_index, 'Train Accuracy'] = round(lgr.score(X_train, Y_train), 2)
MLA_compare.loc[row_index, 'Test Accuracy'] = round(lgr.score(X_test, Y_test), 2)
MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, Y_pred_lgr),2)
MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, Y_pred_lgr),2)
MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, Y_pred_lgr),2)
row_index+=1

#%%
#printing the compare table 

print("\n------------Model Comparison------------\n")
MLA_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True) 
print(MLA_compare,"\n")

#%%
#plotting comaprison graph

fig1 = plt.figure(figsize = (12,10))
fig2 = plt.figure(figsize = (12,10))
fig3 = plt.figure(figsize = (12,10))
fig4 = plt.figure(figsize = (12,10))
fig5 = plt.figure(figsize = (12,10))

ax1 = fig1.add_subplot(2,2,1)
ax2 = fig2.add_subplot(2,2,1)
ax3 = fig3.add_subplot(2,2,1)
ax4 = fig4.add_subplot(2,2,1)
ax5 = fig5.add_subplot(2,2,1)

#test accuracy
ax1.bar(range(0,5),MLA_compare['Test Accuracy'],align='center');
ax1.set_xticks(range(5));
ax1.set_xticks(range(5));
ax1.set_ylim(0,1);
ax1.set_ylabel("Accuracy")
ax1.set_title("Test Accuracy Comparison Between Various Models")
ax1.set_xticklabels(MLA_compare['Model']);

#train accuracy
ax2.bar(range(0,5),MLA_compare['Train Accuracy'],align='center');
ax2.set_xticks(range(5));
ax2.set_xticks(range(5));
ax2.set_ylim(0,1);
ax2.set_ylabel("Accuracy")
ax2.set_title("Train Accuracy Comparison Between Various Models")
ax2.set_xticklabels(MLA_compare['Model']);

#precission
ax3.bar(range(0,5),MLA_compare['Precision'],align='center');
ax3.set_xticks(range(5));
ax3.set_xticks(range(5));
ax3.set_ylim(0,1);
ax3.set_ylabel("Accuracy")
ax3.set_title("Precesion Between Comparison Various Models")
ax3.set_xticklabels(MLA_compare['Model']);

#recall
ax4.bar(range(0,5),MLA_compare['Recall'],align='center');
ax4.set_xticks(range(5));
ax4.set_xticks(range(5));
ax4.set_ylim(0,1);
ax4.set_ylabel("Accuracy")
ax4.set_title("Recall Between Comparison Various Models")
ax4.set_xticklabels(MLA_compare['Model']);

#f1 score
ax5.bar(range(0,5),MLA_compare['F1 score'],align='center');
ax5.set_xticks(range(5));
ax5.set_xticks(range(5));
ax5.set_ylim(0,1);
ax5.set_ylabel("Accuracy")
ax5.set_title("F1 score Comparison Between Various Models")
ax5.set_xticklabels(MLA_compare['Model']);

#%%
#selecting best features

print("\n------------10 Best Features------------\n")
from sklearn.feature_selection import SelectKBest,mutual_info_classif
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)
fit = bestfeatures.fit(X,target,)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns) 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score'] 
print(featureScores.nlargest(10,'Score')) 
