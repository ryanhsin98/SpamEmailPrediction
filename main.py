# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:59:33 2022

@author: ryan8
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#匯入資料
data = pd.read_csv('dataset.csv')
data

#刪除bounce rate或complaint rate等異常值為1
mask1 = data['bounced'] != 1
data = data[(mask1)]
mask2 = data['complaint'] != 1
data = data[(mask2)]
mask3 = data['unsubscribed'] != 1
data = data[(mask3)]

#主動分類label
data['complaint'].describe()
#閥值取 0.074537

#製造label
data['spam_label'] = data['complaint'] >  0.074537
data['spam_label'] = np.where(data['complaint'] >  0.074537, 1 ,\
                              np.where(data['complaint'] <=  0.074537, 0,0))

data['spam_label'].value_counts()

#%%
'''
NLP
'''

x = data['title']
y = data['spam_label']

#stopwords
stpwrdlst = stopwords.words('english') 

cv = CountVectorizer(lowercase=False, stop_words=stpwrdlst)
x_vector = cv.fit_transform(x)
print(cv.get_feature_names()) 

#%%
'''
SMOTE
'''

#樣本平衡
from imblearn.over_sampling import SMOTE
smote = SMOTE()
               
x_smote, y_smote = smote.fit_resample( x_vector, y )

#%%
'''
SVM model
'''

x_train, x_test,y_train, y_test = train_test_split(x_smote,y_smote,test_size = 0.2)

#Grid search
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear']}  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(x_train, y_train)

print(grid.best_params_) 
#{'C': 10, 'gamma': 1, 'kernel': 'rbf'}

best_model = grid.best_estimator_ 
grid_predictions = grid.predict(x_test)
print(classification_report(y_test, grid_predictions))

#5fold
clf = svm.SVC(C=5, gamma=1, kernel='rbf')

#estimate the accuracy
scores = cross_val_score(clf, x_smote, y_smote, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#f1 score
scores = cross_val_score(clf, x_smote, y_smote, cv=5, scoring='f1_macro')
scores
print("%0.2f f1-score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%%
'''
LOGISTIC REGRESSION
'''

#Grid search
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid, cv=10)
logreg_cv.fit(x_smote,y_smote)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#tuned hpyerparameters :(best parameters)  {'C': 0.1, 'penalty': 'l2'}
print("accuracy :",logreg_cv.best_score_)

#Logistic regression
best_logreg=LogisticRegression(C=0.1, penalty='l2')

#5fold
#estimate the accuracy
scores = cross_val_score(best_logreg, x_smote, y_smote, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#f1 score
scores = cross_val_score(best_logreg, x_smote, y_smote, cv=5, scoring='f1_macro')
scores
print("%0.2f f1-score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
