# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:43:06 2021

@author: Youyang Shen
"""

import scipy as sp
# import rasterTools as rt
from sklearn.preprocessing import StandardScaler
# import icm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import tifffile
import msi
import os
import numpy as np

path = r'..\Images'
arr = [os.path.join(r,file) for r,d,f in os.walk(path) for file in f]

path2 = r'C:\Users\Youyang Shen\Documents\Videometer\VideometerLab\Images'
arr2 = [os.path.join(r,file) for r,d,f in os.walk(path2) for file in f]


X =[]
y =[]
for i in range(len(arr)):
    image = msi.read(arr[i])
    label = msi.read(arr2[i])
    X.append(image.pixel_values)
    y.append(label.pixel_values)

X = np.array(X)
y = np.array(y)

X = X.reshape(-1,5)
y = y.reshape(-1,1)
[n,m] = y.shape
sc = StandardScaler()
X = sc.fit_transform(X)
X = X.reshape(n,-1)
X_train = X[:,0:3]
y_train = y[:,0:3]
X_test = X[:,3:5]
y_test = y[:,3:5]

y_train.shape=(y_train.size,) 
cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(SVC(), param_grid=dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)), cv=cv,n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

clf.probability= True
clf.fit(X_train,y_train)   

yp = clf.predict(X_test).reshape(y_test.shape)
print (f1_score(y_test,yp,average='weighted'))




# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape=(h*w,b)

# Get the training set
X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')

# Scale the data
sc = StandardScaler()
X = sc.fit_transform(X)
im = sc.transform(im)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,random_state=0,stratify=y)

y_train.shape=(y_train.size,)    
cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(SVC(), param_grid=dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)), cv=cv,n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

clf.probability= True
clf.fit(X_train,y_train)

yp = clf.predict(X_test).reshape(y_test.shape)
print (f1_score(y_test,yp,average='weighted'))

del X_train, X_test, y_train, y_test

# Predict the whole image and the probability map
labels = clf.predict(im).reshape(h,w)
proba = -clf.predict_log_proba(im).reshape(h,w,y.max())

rt.write_data('../Data/proba_university_svm_proba.tif',proba,GeoT,Proj)
rt.write_data('../Data/proba_university_svm_labels.tif',labels,GeoT,Proj)

# Run ICM
diff = icm.fit(proba,labels,beta=1.25,th=0.01)
print diff
rt.write_data('../Data/tm_university_svm_mrf.tif',labels,GeoT,Proj)