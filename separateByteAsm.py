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


def separateByteAsm():
	#separating byte files and asm files 

	source = 'train'
	destination = 'byteFiles'

	# we will check if the folder 'byteFiles' exists if it not there we will create a folder with the same name
	if not os.path.isdir(destination):
	    os.makedirs(destination)

	# if we have folder called 'train' (train folder contains both .asm files and .bytes files) we will rename it 'asmFiles'
	# for every file that we have in our 'asmFiles' directory we check if it is ending with .bytes, if yes we will move it to
	# 'byteFiles' folder

	# so by the end of this snippet we will separate all the .byte files and .asm files
	if os.path.isdir(source):
	    os.rename(source,'asmFiles')
	    source='asmFiles'
	    data_files = os.listdir(source)
	    for _, _, file in os.walk(source):
	        for f in file:
	            if f.endswith("bytes"):
	                shutil.move(source+'/'+f,destination)



	Y=pd.read_csv("trainLabels.csv")
	total = len(Y)*1.
	ax=sns.countplot(x="Class", data=Y)
	for p in ax.patches:
	        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

	#put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
	ax.yaxis.set_ticks(np.linspace(0, total, 11))

	#adjust the ticklabel to the desired format, without changing the position of the ticks. 
	ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
	plt.show()


