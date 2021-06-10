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



def featureExtraction():
    #intially create five folders
    #first 
    #second
    #thrid
    #fourth
    #fifth
    #this code tells us about random split of files into five folders

    folder_1 ='first'
    folder_2 ='second'
    folder_3 ='third'
    folder_4 ='fourth'
    folder_5 ='fifth'
    folder_6 = 'output'
    for i in [folder_1,folder_2,folder_3,folder_4,folder_5,folder_6]:
        if not os.path.isdir(i):
            os.makedirs(i)

    source='train/'
    files = os.listdir('train')
    ID=df['Id'].tolist()
    data=range(0,10868)
    r.shuffle(data)
    count=0
    for i in range(0,10868):
        if i % 5==0:
            shutil.move(source+files[data[i]],'first')
        elif i%5==1:
            shutil.move(source+files[data[i]],'second')
        elif i%5 ==2:
            shutil.move(source+files[data[i]],'thrid')
        elif i%5 ==3:
            shutil.move(source+files[data[i]],'fourth')
        elif i%5==4:
            shutil.move(source+files[data[i]],'fifth')



    def firstprocess():
        #The prefixes tells about the segments that are present in the asm files
        #There are 450 segments(approx) present in all asm files.
        #this prefixes are best segments that gives us best values.
        
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:','.BSS:','.CODE']
        #this are opcodes that are used to get best results
        
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']
        
        #best keywords that are taken from different blogs
        keywords = ['.dll','std::',':dword']
        
        #Below taken registers are general purpose registers and special registers
        #All the registers which are taken are best 
        registers=['edx','esi','eax','ebx','ecx','edi','ebp','esp','eip']
        file1=open("output\asmsmallfile.txt","w+")
        files = os.listdir('first')
        
        for f in files:
            #filling the values with zeros into the arrays
            prefixescount=np.zeros(len(prefixes),dtype=int)
            opcodescount=np.zeros(len(opcodes),dtype=int)
            keywordcount=np.zeros(len(keywords),dtype=int)
            registerscount=np.zeros(len(registers),dtype=int)
            features=[]
            f2=f.split('.')[0]
            file1.write(f2+",")
            opcodefile.write(f2+" ")
            with codecs.open('first/'+f,encoding='cp1252',errors ='replace') as fli:
                for lines in fli:
                    line=lines.rstrip().split()
                    l=line[0]
                    #counting the prefixs in each and every line
                    for i in range(len(prefixes)):
                        if prefixes[i] in line[0]:
                            prefixescount[i]+=1
                    line=line[1:]
                    #counting the opcodes in each and every line
                    for i in range(len(opcodes)):
                        if any(opcodes[i]==li for li in line):
                            features.append(opcodes[i])
                            opcodescount[i]+=1
                    #counting registers in the line
                    for i in range(len(registers)):
                        for li in line:
                            # we will use registers only in 'text' and 'CODE' segments
                            if registers[i] in li and ('text' in l or 'CODE' in l):
                                registerscount[i]+=1
                    #counting keywords in the line
                    for i in range(len(keywords)):
                        for li in line:
                            if keywords[i] in li:
                                keywordcount[i]+=1
            #pushing the values into the file after reading whole file
            for prefix in prefixescount:
                file1.write(str(prefix)+",")
            for opcode in opcodescount:
                file1.write(str(opcode)+",")
            for register in registerscount:
                file1.write(str(register)+",")
            for key in keywordcount:
                file1.write(str(key)+",")
            file1.write("\n")
        file1.close()


    #same as above 
    def secondprocess():
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:','.BSS:','.CODE']
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']
        keywords = ['.dll','std::',':dword']
        registers=['edx','esi','eax','ebx','ecx','edi','ebp','esp','eip']
        file1=open("output\mediumasmfile.txt","w+")
        files = os.listdir('second')
        for f in files:
            prefixescount=np.zeros(len(prefixes),dtype=int)
            opcodescount=np.zeros(len(opcodes),dtype=int)
            keywordcount=np.zeros(len(keywords),dtype=int)
            registerscount=np.zeros(len(registers),dtype=int)
            features=[]
            f2=f.split('.')[0]
            file1.write(f2+",")
            opcodefile.write(f2+" ")
            with codecs.open('second/'+f,encoding='cp1252',errors ='replace') as fli:
                for lines in fli:
                    line=lines.rstrip().split()
                    l=line[0]
                    for i in range(len(prefixes)):
                        if prefixes[i] in line[0]:
                            prefixescount[i]+=1
                    line=line[1:]
                    for i in range(len(opcodes)):
                        if any(opcodes[i]==li for li in line):
                            features.append(opcodes[i])
                            opcodescount[i]+=1
                    for i in range(len(registers)):
                        for li in line:
                            if registers[i] in li and ('text' in l or 'CODE' in l):
                                registerscount[i]+=1
                    for i in range(len(keywords)):
                        for li in line:
                            if keywords[i] in li:
                                keywordcount[i]+=1
            for prefix in prefixescount:
                file1.write(str(prefix)+",")
            for opcode in opcodescount:
                file1.write(str(opcode)+",")
            for register in registerscount:
                file1.write(str(register)+",")
            for key in keywordcount:
                file1.write(str(key)+",")
            file1.write("\n")
        file1.close()

    # same as smallprocess() functions
    def thirdprocess():
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:','.BSS:','.CODE']
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']
        keywords = ['.dll','std::',':dword']
        registers=['edx','esi','eax','ebx','ecx','edi','ebp','esp','eip']
        file1=open("output\largeasmfile.txt","w+")
        files = os.listdir('thrid')
        for f in files:
            prefixescount=np.zeros(len(prefixes),dtype=int)
            opcodescount=np.zeros(len(opcodes),dtype=int)
            keywordcount=np.zeros(len(keywords),dtype=int)
            registerscount=np.zeros(len(registers),dtype=int)
            features=[]
            f2=f.split('.')[0]
            file1.write(f2+",")
            opcodefile.write(f2+" ")
            with codecs.open('thrid/'+f,encoding='cp1252',errors ='replace') as fli:
                for lines in fli:
                    line=lines.rstrip().split()
                    l=line[0]
                    for i in range(len(prefixes)):
                        if prefixes[i] in line[0]:
                            prefixescount[i]+=1
                    line=line[1:]
                    for i in range(len(opcodes)):
                        if any(opcodes[i]==li for li in line):
                            features.append(opcodes[i])
                            opcodescount[i]+=1
                    for i in range(len(registers)):
                        for li in line:
                            if registers[i] in li and ('text' in l or 'CODE' in l):
                                registerscount[i]+=1
                    for i in range(len(keywords)):
                        for li in line:
                            if keywords[i] in li:
                                keywordcount[i]+=1
            for prefix in prefixescount:
                file1.write(str(prefix)+",")
            for opcode in opcodescount:
                file1.write(str(opcode)+",")
            for register in registerscount:
                file1.write(str(register)+",")
            for key in keywordcount:
                file1.write(str(key)+",")
            file1.write("\n")
        file1.close()


    def fourthprocess():
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:','.BSS:','.CODE']
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']
        keywords = ['.dll','std::',':dword']
        registers=['edx','esi','eax','ebx','ecx','edi','ebp','esp','eip']
        file1=open("output\hugeasmfile.txt","w+")
        files = os.listdir('fourth/')
        for f in files:
            prefixescount=np.zeros(len(prefixes),dtype=int)
            opcodescount=np.zeros(len(opcodes),dtype=int)
            keywordcount=np.zeros(len(keywords),dtype=int)
            registerscount=np.zeros(len(registers),dtype=int)
            features=[]
            f2=f.split('.')[0]
            file1.write(f2+",")
            opcodefile.write(f2+" ")
            with codecs.open('fourth/'+f,encoding='cp1252',errors ='replace') as fli:
                for lines in fli:
                    line=lines.rstrip().split()
                    l=line[0]
                    for i in range(len(prefixes)):
                        if prefixes[i] in line[0]:
                            prefixescount[i]+=1
                    line=line[1:]
                    for i in range(len(opcodes)):
                        if any(opcodes[i]==li for li in line):
                            features.append(opcodes[i])
                            opcodescount[i]+=1
                    for i in range(len(registers)):
                        for li in line:
                            if registers[i] in li and ('text' in l or 'CODE' in l):
                                registerscount[i]+=1
                    for i in range(len(keywords)):
                        for li in line:
                            if keywords[i] in li:
                                keywordcount[i]+=1
            for prefix in prefixescount:
                file1.write(str(prefix)+",")
            for opcode in opcodescount:
                file1.write(str(opcode)+",")
            for register in registerscount:
                file1.write(str(register)+",")
            for key in keywordcount:
                file1.write(str(key)+",")
            file1.write("\n")
        file1.close()


    def fifthprocess():
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:','.BSS:','.CODE']
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']
        keywords = ['.dll','std::',':dword']
        registers=['edx','esi','eax','ebx','ecx','edi','ebp','esp','eip']
        file1=open("output\trainasmfile.txt","w+")
        files = os.listdir('fifth/')
        for f in files:
            prefixescount=np.zeros(len(prefixes),dtype=int)
            opcodescount=np.zeros(len(opcodes),dtype=int)
            keywordcount=np.zeros(len(keywords),dtype=int)
            registerscount=np.zeros(len(registers),dtype=int)
            features=[]
            f2=f.split('.')[0]
            file1.write(f2+",")
            opcodefile.write(f2+" ")
            with codecs.open('fifth/'+f,encoding='cp1252',errors ='replace') as fli:
                for lines in fli:
                    line=lines.rstrip().split()
                    l=line[0]
                    for i in range(len(prefixes)):
                        if prefixes[i] in line[0]:
                            prefixescount[i]+=1
                    line=line[1:]
                    for i in range(len(opcodes)):
                        if any(opcodes[i]==li for li in line):
                            features.append(opcodes[i])
                            opcodescount[i]+=1
                    for i in range(len(registers)):
                        for li in line:
                            if registers[i] in li and ('text' in l or 'CODE' in l):
                                registerscount[i]+=1
                    for i in range(len(keywords)):
                        for li in line:
                            if keywords[i] in li:
                                keywordcount[i]+=1
            for prefix in prefixescount:
                file1.write(str(prefix)+",")
            for opcode in opcodescount:
                file1.write(str(opcode)+",")
            for register in registerscount:
                file1.write(str(register)+",")
            for key in keywordcount:
                file1.write(str(key)+",")
            file1.write("\n")
        file1.close()


    def main():
        #the below code is used for multiprogramming
        #the number of process depends upon the number of cores present System
        #process is used to call multiprogramming
        manager=multiprocessing.Manager()   
        p1=Process(target=firstprocess)
        p2=Process(target=secondprocess)
        p3=Process(target=thirdprocess)
        p4=Process(target=fourthprocess)
        p5=Process(target=fifthprocess)
        #p1.start() is used to start the thread execution
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        #After completion all the threads are joined
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()

    # if __name__=="__main__":
    main()


    # asmoutputfile.csv(output genarated from the above two cells) will contain all the extracted features from .asm files
    # this file will be uploaded in the drive, you can directly use this
    dfasm=pd.read_csv("asmoutputfile.csv")
    Y.columns = ['ID', 'Class']
    result_asm = pd.merge(dfasm, Y,on='ID', how='left')
    result_asm.head()


    #file sizes of byte files

    files=os.listdir('asmFiles')
    filenames=Y['ID'].tolist()
    class_y=Y['Class'].tolist()
    class_bytes=[]
    sizebytes=[]
    fnames=[]
    for file in files:
        statinfo=os.stat('asmFiles/'+file)
        # split the file name at '.' and take the first part of it i.e the file name
        file=file.split('.')[0]
        if any(file == filename for filename in filenames):
            i=filenames.index(file)
            class_bytes.append(class_y[i])
            # converting into Mb's
            sizebytes.append(statinfo.st_size/(1024.0*1024.0))
            fnames.append(file)
    asm_size_byte=pd.DataFrame({'ID':fnames,'size':sizebytes,'Class':class_bytes})
    print (asm_size_byte.head())


    #boxplot of asm files
    ax = sns.boxplot(x="Class", y="size", data=asm_size_byte)
    plt.title("boxplot of .bytes file sizes")
    plt.show()


    # add the file size feature to previous extracted features
    print(result_asm.shape)
    print(asm_size_byte.shape)
    result_asm = pd.merge(result_asm, asm_size_byte.drop(['Class'], axis=1),on='ID', how='left')
    result_asm.head()

    # we normalize the data each column 
    result_asm = normalize(result_asm)
    result_asm.head()

    ax = sns.boxplot(x="Class", y=".text:", data=result_asm)
    plt.title("boxplot of .asm text segment")
    plt.show()

    ax = sns.boxplot(x="Class", y=".Pav:", data=result_asm)
    plt.title("boxplot of .asm pav segment")
    plt.show()

    ax = sns.boxplot(x="Class", y=".data:", data=result_asm)
    plt.title("boxplot of .asm data segment")
    plt.show()

    ax = sns.boxplot(x="Class", y=".bss:", data=result_asm)
    plt.title("boxplot of .asm bss segment")
    plt.show()

    ax = sns.boxplot(x="Class", y=".rdata:", data=result_asm)
    plt.title("boxplot of .asm rdata segment")
    plt.show()

    ax = sns.boxplot(x="Class", y="jmp", data=result_asm)
    plt.title("boxplot of .asm jmp opcode")
    plt.show()

    ax = sns.boxplot(x="Class", y="mov", data=result_asm)
    plt.title("boxplot of .asm mov opcode")
    plt.show()

    ax = sns.boxplot(x="Class", y="retf", data=result_asm)
    plt.title("boxplot of .asm retf opcode")
    plt.show()

    ax = sns.boxplot(x="Class", y="push", data=result_asm)
    plt.title("boxplot of .asm push opcode")
    plt.show()

    #multivariate analysis on byte files
    #this is with perplexity 50
    xtsne=TSNE(perplexity=50)
    results=xtsne.fit_transform(result_asm.drop(['ID','Class'], axis=1).fillna(0))
    vis_x = results[:, 0]
    vis_y = results[:, 1   ]
    plt.scatter(vis_x, vis_y, c=data_y, cmap=plt.cm.get_cmap("jet", 9))
    plt.colorbar(ticks=range(10))
    plt.clim(0.5, 9)
    plt.show()

    # by univariate analysis on the .asm file features we are getting very negligible information from 
    # 'rtn', '.BSS:' '.CODE' features, so heare we are trying multivariate analysis after removing those features
    # the plot looks very messy

    xtsne=TSNE(perplexity=30)
    results=xtsne.fit_transform(result_asm.drop(['ID','Class', 'rtn', '.BSS:', '.CODE','size'], axis=1))
    vis_x = results[:, 0]
    vis_y = results[:, 1]
    plt.scatter(vis_x, vis_y, c=data_y, cmap=plt.cm.get_cmap("jet", 9))
    plt.colorbar(ticks=range(10))
    plt.clim(0.5, 9)
    plt.show()