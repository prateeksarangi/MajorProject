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

def RandomModel():
	# Random Model

	# we need to generate 9 numbers and the sum of numbers should be 1
	# one solution is to genarate 9 numbers and divide each of the numbers by their sum

	test_data_len = X_test.shape[0]
	cv_data_len = X_cv.shape[0]

	# we create a output array that has exactly same size as the CV data
	cv_predicted_y = np.zeros((cv_data_len,9))
	for i in range(cv_data_len):
	    rand_probs = np.random.rand(1,9)
	    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
	print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))


	# Test-Set error.
	#we create a output array that has exactly same as the test data
	test_predicted_y = np.zeros((test_data_len,9))
	for i in range(test_data_len):
	    rand_probs = np.random.rand(1,9)
	    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
	print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))

	predicted_y =np.argmax(test_predicted_y, axis=1)
	plot_confusion_matrix(y_test, predicted_y+1)




# kNN
def kNN_Model():
	alpha = [x for x in range(1, 15, 2)]
	cv_log_error_array=[]
	for i in alpha:
	    k_cfl=KNeighborsClassifier(n_neighbors=i)
	    k_cfl.fit(X_train,y_train)
	    sig_clf = CalibratedClassifierCV(k_cfl, method="sigmoid")
	    sig_clf.fit(X_train, y_train)
	    predict_y = sig_clf.predict_proba(X_cv)
	    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=k_cfl.classes_, eps=1e-15))
	    
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
	k_cfl.fit(X_train,y_train)
	sig_clf = CalibratedClassifierCV(k_cfl, method="sigmoid")
	sig_clf.fit(X_train, y_train)
	    
	predict_y = sig_clf.predict_proba(X_train)
	print ('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y))
	predict_y = sig_clf.predict_proba(X_cv)
	print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y))
	predict_y = sig_clf.predict_proba(X_test)
	print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y))
	plot_confusion_matrix(y_test, sig_clf.predict(X_test))



# LR
def LR_Model():
	alpha = [10 ** x for x in range(-5, 4)]
	cv_log_error_array=[]
	for i in alpha:
	    logisticR=LogisticRegression(penalty='l2',C=i,class_weight='balanced')
	    logisticR.fit(X_train,y_train)
	    sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
	    sig_clf.fit(X_train, y_train)
	    predict_y = sig_clf.predict_proba(X_cv)
	    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=logisticR.classes_, eps=1e-15))
	    
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
	logisticR.fit(X_train,y_train)
	sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
	sig_clf.fit(X_train, y_train)
	pred_y=sig_clf.predict(X_test)

	predict_y = sig_clf.predict_proba(X_train)
	print ('log loss for train data',log_loss(y_train, predict_y, labels=logisticR.classes_, eps=1e-15))
	predict_y = sig_clf.predict_proba(X_cv)
	print ('log loss for cv data',log_loss(y_cv, predict_y, labels=logisticR.classes_, eps=1e-15))
	predict_y = sig_clf.predict_proba(X_test)
	print ('log loss for test data',log_loss(y_test, predict_y, labels=logisticR.classes_, eps=1e-15))
	plot_confusion_matrix(y_test, sig_clf.predict(X_test))



# RF
def RF_Model():
	alpha=[10,50,100,500,1000,2000,3000]
	cv_log_error_array=[]
	train_log_error_array=[]
	from sklearn.ensemble import RandomForestClassifier
	for i in alpha:
	    r_cfl=RandomForestClassifier(n_estimators=i,random_state=42,n_jobs=-1)
	    r_cfl.fit(X_train,y_train)
	    sig_clf = CalibratedClassifierCV(r_cfl, method="sigmoid")
	    sig_clf.fit(X_train, y_train)
	    predict_y = sig_clf.predict_proba(X_cv)
	    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=r_cfl.classes_, eps=1e-15))

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
	r_cfl.fit(X_train,y_train)
	sig_clf = CalibratedClassifierCV(r_cfl, method="sigmoid")
	sig_clf.fit(X_train, y_train)

	predict_y = sig_clf.predict_proba(X_train)
	print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y))
	predict_y = sig_clf.predict_proba(X_cv)
	print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y))
	predict_y = sig_clf.predict_proba(X_test)
	print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y))
	plot_confusion_matrix(y_test, sig_clf.predict(X_test))



# XgBoost
def XgBoostModel():
	alpha=[10,50,100,500,1000,2000]
	cv_log_error_array=[]
	for i in alpha:
	    x_cfl=XGBClassifier(n_estimators=i,nthread=-1)
	    x_cfl.fit(X_train,y_train)
	    sig_clf = CalibratedClassifierCV(x_cfl, method="sigmoid")
	    sig_clf.fit(X_train, y_train)
	    predict_y = sig_clf.predict_proba(X_cv)
	    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=x_cfl.classes_, eps=1e-15))

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
	x_cfl.fit(X_train,y_train)
	sig_clf = CalibratedClassifierCV(x_cfl, method="sigmoid")
	sig_clf.fit(X_train, y_train)
	    
	predict_y = sig_clf.predict_proba(X_train)
	print ('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y))
	predict_y = sig_clf.predict_proba(X_cv)
	print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y))
	predict_y = sig_clf.predict_proba(X_test)
	print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y))
	plot_confusion_matrix(y_test, sig_clf.predict(X_test))



# XgBoost with Random search
def XgBoostBest_Model():
	x_cfl=XGBClassifier()

	prams={
	    'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
	     'n_estimators':[100,200,500,1000,2000],
	     'max_depth':[3,5,10],
	    'colsample_bytree':[0.1,0.3,0.5,1],
	    'subsample':[0.1,0.3,0.5,1]
	}
	random_cfl1=RandomizedSearchCV(x_cfl,param_distributions=prams,verbose=10,n_jobs=-1,)
	random_cfl1.fit(X_train,y_train)
	print (random_cfl1.best_params_)

	x_cfl=XGBClassifier(n_estimators=2000, learning_rate=0.05, colsample_bytree=1, max_depth=3)
	x_cfl.fit(X_train,y_train)
	c_cfl=CalibratedClassifierCV(x_cfl,method='sigmoid')
	c_cfl.fit(X_train,y_train)

	predict_y = c_cfl.predict_proba(X_train)
	print ('train loss',log_loss(y_train, predict_y))
	predict_y = c_cfl.predict_proba(X_cv)
	print ('cv loss',log_loss(y_cv, predict_y))
	predict_y = c_cfl.predict_proba(X_test)
	print ('test loss',log_loss(y_test, predict_y))