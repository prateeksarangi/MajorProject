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


def featureExtration():
	#file sizes of byte files

	files=os.listdir('byteFiles')
	filenames=Y['Id'].tolist()
	class_y=Y['Class'].tolist()
	class_bytes=[]
	sizebytes=[]
	fnames=[]
	for file in files:
	    statinfo=os.stat('byteFiles/'+file)
	    # split the file name at '.' and take the first part of it i.e the file name
	    file=file.split('.')[0]
	    if any(file == filename for filename in filenames):
	        i=filenames.index(file)
	        class_bytes.append(class_y[i])
	        # converting into Mb's
	        sizebytes.append(statinfo.st_size/(1024.0*1024.0))
	        fnames.append(file)
	data_size_byte=pd.DataFrame({'ID':fnames,'size':sizebytes,'Class':class_bytes})
	print (data_size_byte.head())


	#boxplot of byte files
	ax = sns.boxplot(x="Class", y="size", data=data_size_byte)
	plt.title("boxplot of .bytes file sizes")
	plt.show()


	#removal of addres from byte files
	# contents of .byte files
	# ----------------
	#00401000 56 8D 44 24 08 50 8B F1 E8 1C 1B 00 00 C7 06 08 
	#-------------------
	#we remove the starting address 00401000

	files = os.listdir('byteFiles')
	filenames=[]

	array=[]
	for file in files:
	    if(f.endswith("bytes")):
	        file=file.split('.')[0]
	        text_file = open('byteFiles/'+file+".txt", 'w+')
	        with open('byteFiles/'+file+'.bytes',"r") as fp:
	            lines=""
	            for line in fp:
	                a=line.rstrip().split(" ")[1:]
	                b=' '.join(a)
	                b=b+"\n"
	                text_file.write(b)
	            fp.close()
	            os.remove('byteFiles/'+file+'.bytes')
	        text_file.close()

	files = os.listdir('byteFiles')
	filenames2=[]
	feature_matrix = np.zeros((len(files),257),dtype=int)
	k=0


	#program to convert into bag of words of bytefiles
	#this is custom-built bag of words this is unigram bag of words
	byte_feature_file=open('result.csv','w+')
	byte_feature_file.write("ID,0,1,2,3,4,5,6,7,8,9,0a,0b,0c,0d,0e,0f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f,20,21,22,23,24,25,26,27,28,29,2a,2b,2c,2d,2e,2f,30,31,32,33,34,35,36,37,38,39,3a,3b,3c,3d,3e,3f,40,41,42,43,44,45,46,47,48,49,4a,4b,4c,4d,4e,4f,50,51,52,53,54,55,56,57,58,59,5a,5b,5c,5d,5e,5f,60,61,62,63,64,65,66,67,68,69,6a,6b,6c,6d,6e,6f,70,71,72,73,74,75,76,77,78,79,7a,7b,7c,7d,7e,7f,80,81,82,83,84,85,86,87,88,89,8a,8b,8c,8d,8e,8f,90,91,92,93,94,95,96,97,98,99,9a,9b,9c,9d,9e,9f,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aa,ab,ac,ad,ae,af,b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,ba,bb,bc,bd,be,bf,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,ca,cb,cc,cd,ce,cf,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db,dc,dd,de,df,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,ea,eb,ec,ed,ee,ef,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,fa,fb,fc,fd,fe,ff,??")
	for file in files:
	    filenames2.append(f)
	    byte_feature_file.write(file+",")
	    if(file.endswith("txt")):
	        with open('byteFiles/'+file,"r") as byte_flie:
	            for lines in byte_flie:
	                line=lines.rstrip().split(" ")
	                for hex_code in line:
	                    if hex_code=='??':
	                        feature_matrix[k][256]+=1
	                    else:
	                        feature_matrix[k][int(hex_code,16)]+=1
	        byte_flie.close()
	    for i in feature_matrix[k]:
	        byte_feature_file.write(str(i)+",")
	    byte_feature_file.write("\n")
	    
	    k += 1

	byte_feature_file.close()


	byte_features=pd.read_csv("result.csv")
	print (byte_features.head())


	def normalize(df):
	    result1 = df.copy()
	    for feature_name in df.columns:
	        if (str(feature_name) != str('ID') and str(feature_name)!=str('Class')):
	            max_value = df[feature_name].max()
	            min_value = df[feature_name].min()
	            result1[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	    return result1
	result = normalize(result)


	data_y = result['Class']
	result.head()


	#multivariate analysis on byte files
	#this is with perplexity 50
	xtsne=TSNE(perplexity=50)
	results=xtsne.fit_transform(result.drop(['ID','Class'], axis=1))
	vis_x = results[:, 0]
	vis_y = results[:, 1]
	plt.scatter(vis_x, vis_y, c=data_y, cmap=plt.cm.get_cmap("jet", 9))
	plt.colorbar(ticks=range(10))
	plt.clim(0.5, 9)
	plt.show()


	#this is with perplexity 30
	xtsne=TSNE(perplexity=30)
	results=xtsne.fit_transform(result.drop(['ID','Class'], axis=1))
	vis_x = results[:, 0]
	vis_y = results[:, 1]
	plt.scatter(vis_x, vis_y, c=data_y, cmap=plt.cm.get_cmap("jet", 9))
	plt.colorbar(ticks=range(10))
	plt.clim(0.5, 9)
	plt.show()


