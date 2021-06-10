import warnings
import shutil
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn import preprocessing
import pandas as pd
from multiprocessing import Process# this is used for multithreading
import multiprocessing
import codecs# this is used for file operations 
import random as r
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from ConfusionMatrixPrint import plot_confusion_matrix

# kNN
def kNN_Model():
	alpha = [x for x in range(1, 21,2)]
	cv_log_error_array=[]
	for i in alpha:
	    k_cfl=KNeighborsClassifier(n_neighbors=i)
	    k_cfl.fit(X_train_asm,y_train_asm)
	    sig_clf = CalibratedClassifierCV(k_cfl, method="sigmoid")
	    sig_clf.fit(X_train_asm, y_train_asm)
	    predict_y = sig_clf.predict_proba(X_cv_asm)
	    cv_log_error_array.append(log_loss(y_cv_asm, predict_y, labels=k_cfl.classes_, eps=1e-15))
	    
	for i in range(len(cv_log_error_array)):
	    print ('log_loss for k = ',alpha[i],'is',cv_log_error_array[i])

	best_alpha = np.argmin(cv_log_error_array)
	    
	fig, ax = plt.subplots()
	ax.plot(alpha, cv_log_error_array,c='g')
	for i, txt in enumerate(np.round(cv_log_error_array,3)):
	    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
	plt.grid()
	plt.title("Cross Validation Error for each alpha")
	plt.xlabel("Alpha i's")
	plt.ylabel("Error measure")
	plt.show()

	k_cfl=KNeighborsClassifier(n_neighbors=alpha[best_alpha])
	k_cfl.fit(X_train_asm,y_train_asm)
	sig_clf = CalibratedClassifierCV(k_cfl, method="sigmoid")
	sig_clf.fit(X_train_asm, y_train_asm)
	pred_y=sig_clf.predict(X_test_asm)


	predict_y = sig_clf.predict_proba(X_train_asm)
	print ('log loss for train data',log_loss(y_train_asm, predict_y))
	predict_y = sig_clf.predict_proba(X_cv_asm)
	print ('log loss for cv data',log_loss(y_cv_asm, predict_y))
	predict_y = sig_clf.predict_proba(X_test_asm)
	print ('log loss for test data',log_loss(y_test_asm, predict_y))
	plot_confusion_matrix(y_test_asm,sig_clf.predict(X_test_asm))


# LR
def LR_Model():
	alpha = [10 ** x for x in range(-5, 4)]
	cv_log_error_array=[]
	for i in alpha:
	    logisticR=LogisticRegression(penalty='l2',C=i,class_weight='balanced')
	    logisticR.fit(X_train_asm,y_train_asm)
	    sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
	    sig_clf.fit(X_train_asm, y_train_asm)
	    predict_y = sig_clf.predict_proba(X_cv_asm)
	    cv_log_error_array.append(log_loss(y_cv_asm, predict_y, labels=logisticR.classes_, eps=1e-15))
	    
	for i in range(len(cv_log_error_array)):
	    print ('log_loss for c = ',alpha[i],'is',cv_log_error_array[i])

	best_alpha = np.argmin(cv_log_error_array)
	    
	fig, ax = plt.subplots()
	ax.plot(alpha, cv_log_error_array,c='g')
	for i, txt in enumerate(np.round(cv_log_error_array,3)):
	    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
	plt.grid()
	plt.title("Cross Validation Error for each alpha")
	plt.xlabel("Alpha i's")
	plt.ylabel("Error measure")
	plt.show()

	logisticR=LogisticRegression(penalty='l2',C=alpha[best_alpha],class_weight='balanced')
	logisticR.fit(X_train_asm,y_train_asm)
	sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
	sig_clf.fit(X_train_asm, y_train_asm)

	predict_y = sig_clf.predict_proba(X_train_asm)
	print ('log loss for train data',(log_loss(y_train_asm, predict_y, labels=logisticR.classes_, eps=1e-15)))
	predict_y = sig_clf.predict_proba(X_cv_asm)
	print ('log loss for cv data',(log_loss(y_cv_asm, predict_y, labels=logisticR.classes_, eps=1e-15)))
	predict_y = sig_clf.predict_proba(X_test_asm)
	print ('log loss for test data',(log_loss(y_test_asm, predict_y, labels=logisticR.classes_, eps=1e-15)))
	plot_confusion_matrix(y_test_asm,sig_clf.predict(X_test_asm))


# RF
def RF_Model():
	alpha=[10,50,100,500,1000,2000,3000]
	cv_log_error_array=[]
	for i in alpha:
	    r_cfl=RandomForestClassifier(n_estimators=i,random_state=42,n_jobs=-1)
	    r_cfl.fit(X_train_asm,y_train_asm)
	    sig_clf = CalibratedClassifierCV(r_cfl, method="sigmoid")
	    sig_clf.fit(X_train_asm, y_train_asm)
	    predict_y = sig_clf.predict_proba(X_cv_asm)
	    cv_log_error_array.append(log_loss(y_cv_asm, predict_y, labels=r_cfl.classes_, eps=1e-15))

	for i in range(len(cv_log_error_array)):
	    print ('log_loss for c = ',alpha[i],'is',cv_log_error_array[i])


	best_alpha = np.argmin(cv_log_error_array)

	fig, ax = plt.subplots()
	ax.plot(alpha, cv_log_error_array,c='g')
	for i, txt in enumerate(np.round(cv_log_error_array,3)):
	    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
	plt.grid()
	plt.title("Cross Validation Error for each alpha")
	plt.xlabel("Alpha i's")
	plt.ylabel("Error measure")
	plt.show()

	r_cfl=RandomForestClassifier(n_estimators=alpha[best_alpha],random_state=42,n_jobs=-1)
	r_cfl.fit(X_train_asm,y_train_asm)
	sig_clf = CalibratedClassifierCV(r_cfl, method="sigmoid")
	sig_clf.fit(X_train_asm, y_train_asm)
	predict_y = sig_clf.predict_proba(X_train_asm)
	print ('log loss for train data',(log_loss(y_train_asm, predict_y, labels=sig_clf.classes_, eps=1e-15)))
	predict_y = sig_clf.predict_proba(X_cv_asm)
	print ('log loss for cv data',(log_loss(y_cv_asm, predict_y, labels=sig_clf.classes_, eps=1e-15)))
	predict_y = sig_clf.predict_proba(X_test_asm)
	print ('log loss for test data',(log_loss(y_test_asm, predict_y, labels=sig_clf.classes_, eps=1e-15)))
	plot_confusion_matrix(y_test_asm,sig_clf.predict(X_test_asm))



# XgBoost
def XgBoosr_Model():
	alpha=[10,50,100,500,1000,2000,3000]
	cv_log_error_array=[]
	for i in alpha:
	    x_cfl=XGBClassifier(n_estimators=i,nthread=-1)
	    x_cfl.fit(X_train_asm,y_train_asm)
	    sig_clf = CalibratedClassifierCV(x_cfl, method="sigmoid")
	    sig_clf.fit(X_train_asm, y_train_asm)
	    predict_y = sig_clf.predict_proba(X_cv_asm)
	    cv_log_error_array.append(log_loss(y_cv_asm, predict_y, labels=x_cfl.classes_, eps=1e-15))

	for i in range(len(cv_log_error_array)):
	    print ('log_loss for c = ',alpha[i],'is',cv_log_error_array[i])


	best_alpha = np.argmin(cv_log_error_array)

	fig, ax = plt.subplots()
	ax.plot(alpha, cv_log_error_array,c='g')
	for i, txt in enumerate(np.round(cv_log_error_array,3)):
	    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
	plt.grid()
	plt.title("Cross Validation Error for each alpha")
	plt.xlabel("Alpha i's")
	plt.ylabel("Error measure")
	plt.show()

	x_cfl=XGBClassifier(n_estimators=alpha[best_alpha],nthread=-1)
	x_cfl.fit(X_train_asm,y_train_asm)
	sig_clf = CalibratedClassifierCV(x_cfl, method="sigmoid")
	sig_clf.fit(X_train_asm, y_train_asm)
	    
	predict_y = sig_clf.predict_proba(X_train_asm)

	print ('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train_asm, predict_y))
	predict_y = sig_clf.predict_proba(X_cv_asm)
	print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv_asm, predict_y))
	predict_y = sig_clf.predict_proba(X_test_asm)
	print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test_asm, predict_y))
	plot_confusion_matrix(y_test_asm,sig_clf.predict(X_test_asm))



def XgBoostBest_Model():
	x_cfl=XGBClassifier()

	prams={
	    'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
	     'n_estimators':[100,200,500,1000,2000],
	     'max_depth':[3,5,10],
	    'colsample_bytree':[0.1,0.3,0.5,1],
	    'subsample':[0.1,0.3,0.5,1]
	}
	random_cfl=RandomizedSearchCV(x_cfl,param_distributions=prams,verbose=10,n_jobs=-1,)
	random_cfl.fit(X_train_asm,y_train_asm)

	print (random_cfl.best_params_)

	x_cfl=XGBClassifier(n_estimators=200,subsample=0.5,learning_rate=0.15,colsample_bytree=0.5,max_depth=3)
	x_cfl.fit(X_train_asm,y_train_asm)
	c_cfl=CalibratedClassifierCV(x_cfl,method='sigmoid')
	c_cfl.fit(X_train_asm,y_train_asm)

	predict_y = c_cfl.predict_proba(X_train_asm)
	print ('train loss',log_loss(y_train_asm, predict_y))
	predict_y = c_cfl.predict_proba(X_cv_asm)
	print ('cv loss',log_loss(y_cv_asm, predict_y))
	predict_y = c_cfl.predict_proba(X_test_asm)
	print ('test loss',log_loss(y_test_asm, predict_y))