import pandas as pd
import numpy as np
from util import effortCal,divideIntoKSubset
import warnings
warnings.filterwarnings("ignore")

from supriesd import suprised_Popt_ACC 



cols=["nb","knn","dt","svc","rfc","gbtc","adtc"]
Popt=pd.DataFrame(columns=cols)
ACC=pd.DataFrame(columns=cols)


labels=["bugzilla", "columba", "jdt", "platform", "mozilla", "postgres"]

for label in labels:
    data=pd.read_csv("../input/%s.csv"%label)
    data.drop("transactionid",axis=1,inplace=True)
    data.drop("commitdate",axis=1,inplace=True)
    
    k=10
    dataSubsetList=divideIntoKSubset(k=k,data=data)
    
    
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
         #不知道为什么十折之后有些数据的类型会变成object，所以用这个函数自动推断类型
        train=train.convert_objects(convert_numeric=True) 
        test=test.convert_objects(convert_numeric=True)
        #print(test.shape)
        #print(train.shape)
        #print(train.dtypes)
        
        Popt_temp,ACC_temp=suprised_Popt_ACC(train,test)
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
    #一个项目的10折交叉验证得到的Popt和ACC结果保存文件
    p.to_csv(r"../output/suprised/s_"+label+"_Popt.csv",index=None)
    a.to_csv(r"../output/suprised/s_"+label+"._ACC.csv",index=None)
    
    Popt=pd.concat([Popt,p])
    ACC=pd.concat([ACC,a])
    
Popt.to_csv(r"../output/suprised/s_Popt.csv",index=None)
ACC.to_csv(r"../output/suprised/s_ACC.csv",index=None)

    



