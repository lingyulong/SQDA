import pandas as pd
import numpy as np
from util import effortCal,divideIntoKSubset
import warnings
warnings.filterwarnings("ignore")

from supriesd import suprised_Popt_ACC 

data=pd.read_csv(r"../input/bugzilla.csv")
data.drop("transactionid",axis=1,inplace=True)
data.drop("commitdate",axis=1,inplace=True)
#print(data.dtypes)
#print("*"*50)

k=10
dataSubsetList=divideIntoKSubset(k=k,data=data)


Popt={}
ACC={}

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
    Popt[i]=Popt_temp
    ACC[i]=ACC_temp
    print(Popt_temp)
    print(ACC_temp)
    print("*"*50)
    
    
  
print("*"*50)
print(Popt)
print()

print(ACC)

p=pd.DataFrame(Popt).T
a=pd.DataFrame(ACC).T

p.to_csv(r"../output/suprised/s_bugzilla_Popt.csv",index=None)
a.to_csv(r"../output/suprised/s_bugzilla_ACC.csv",index=None)

