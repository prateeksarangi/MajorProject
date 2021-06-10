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


def test_train_Split():
	data_y = result['Class']
	# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
	X_train, X_test, y_train, y_test = train_test_split(result.drop(['ID','Class'], axis=1), data_y,stratify=data_y,test_size=0.20)
	# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
	X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train,stratify=y_train,test_size=0.20)


	print('Number of data points in train data:', X_train.shape[0])
	print('Number of data points in test data:', X_test.shape[0])
	print('Number of data points in cross validation data:', X_cv.shape[0])


	# it returns a dict, keys as class labels and values as the number of data points in that class
	train_class_distribution = y_train.value_counts().sortlevel()
	test_class_distribution = y_test.value_counts().sortlevel()
	cv_class_distribution = y_cv.value_counts().sortlevel()

	my_colors = 'rgbkymc'
	train_class_distribution.plot(kind='bar', color=my_colors)
	plt.xlabel('Class')
	plt.ylabel('Data points per Class')
	plt.title('Distribution of yi in train data')
	plt.grid()
	plt.show()

	# -(train_class_distribution.values): the minus sign will give us in decreasing order
	sorted_yi = np.argsort(-train_class_distribution.values)
	for i in sorted_yi:
	    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/y_train.shape[0]*100), 3), '%)')

	    
	print('-'*80)
	my_colors = 'rgbkymc'
	test_class_distribution.plot(kind='bar', color=my_colors)
	plt.xlabel('Class')
	plt.ylabel('Data points per Class')
	plt.title('Distribution of yi in test data')
	plt.grid()
	plt.show()

	# -(train_class_distribution.values): the minus sign will give us in decreasing order
	sorted_yi = np.argsort(-test_class_distribution.values)
	for i in sorted_yi:
	    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/y_test.shape[0]*100), 3), '%)')

	print('-'*80)
	my_colors = 'rgbkymc'
	cv_class_distribution.plot(kind='bar', color=my_colors)
	plt.xlabel('Class')
	plt.ylabel('Data points per Class')
	plt.title('Distribution of yi in cross validation data')
	plt.grid()
	plt.show()

	# -(train_class_distribution.values): the minus sign will give us in decreasing order
	sorted_yi = np.argsort(-train_class_distribution.values)
	for i in sorted_yi:
	    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/y_cv.shape[0]*100), 3), '%)')



