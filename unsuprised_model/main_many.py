import pandas as pd
import numpy as np
from util import effortCal,divideIntoKSubset
from unsupriesd import unsuprised_Popt_ACC 

cols=["ns","nm","nf","entropy","lt","fix","ndev","pd","nuc","exp","rexp","sexp"]
Popt=pd.DataFrame(columns=cols)
ACC=pd.DataFrame(columns=cols)


labels=["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]

for label in labels:
    data=pd.read_csv(r"../input/%s.csv"%label)
    data.drop("transactionid",axis=1,inplace=True)
    data.drop("commitdate",axis=1,inplace=True)
    
    k=10
    dataSubsetList=divideIntoKSubset(k=k,data=data)
    #print(dataSubsetList[0].describe())
    #print(type(dataSubsetList[0]))
    
    
    Popt_project={}
    ACC_project={}
    
    for i in range(k):
        train=pd.DataFrame(columns=list(data))
        test=pd.DataFrame(columns=list(data))
        test=dataSubsetList[i]
        for j in range(k):
            if(i!=j):
                train=pd.concat([train,dataSubsetList[j]])
        
        print(i)
        print(test.shape)
        print(train.shape)
        #unsuprised_Popt_ACC(train,test)
        #print("*"*50)
    
        Popt_temp,ACC_temp=unsuprised_Popt_ACC(train,test)
        Popt_project[i]=Popt_temp
        ACC_project[i]=ACC_temp
        #print(Popt_temp)
        #print(ACC_temp)
        #print("*"*50)
      
    print("*"*50)
    print(ACC)
    

    p=pd.DataFrame(Popt_project).T
    p=p[cols]
    
    a=pd.DataFrame(ACC_project).T
    a=a[cols]
    
    p.to_csv(r"../output/unsuprised/us_"+label+"_Popt.csv",index=None)
    a.to_csv(r"../output/unsuprised/us_"+label+"._ACC.csv",index=None)
    
    Popt=pd.concat([Popt,p])
    ACC=pd.concat([ACC,a])
    
Popt.to_csv(r"../output/unsuprised/us_Popt.csv",index=None)
ACC.to_csv(r"../output/unsuprised/us_ACC.csv",index=None)

    



