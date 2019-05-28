import pandas as pd
import numpy as np
from util import effortCal,divideIntoKSubset
from unsupriesd import unsuprised_Popt_ACC 

data=pd.read_csv(r"../input/bugzilla.csv")
data.drop("transactionid",axis=1,inplace=True)
data.drop("commitdate",axis=1,inplace=True)

k=10
dataSubsetList=divideIntoKSubset(k=k,data=data)
#print(dataSubsetList[0].describe())
#print(type(dataSubsetList[0]))


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
    print(test.shape)
    print(train.shape)
    #unsuprised_Popt_ACC(train,test)
    #print("*"*50)

    Popt_temp,ACC_temp=unsuprised_Popt_ACC(train,test)
    Popt[i]=Popt_temp
    ACC[i]=ACC_temp
    #print(Popt_temp)
    #print(ACC_temp)
    #print("*"*50)
  
print("*"*50)
print(ACC)

p=pd.DataFrame(Popt).T
a=pd.DataFrame(ACC).T
p.to_csv(r"../output/unsuprised/us_bugzilla_Popt.csv",index=None)
a.to_csv(r"../output/unsuprised/us_bugzilla_ACC.csv",index=None)

    



