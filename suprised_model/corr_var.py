import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from util import effortCal,computePopt,computeACC,doSample

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
train=pd.read_csv(r"../input/bugzilla.csv")
train.drop("transactionid",axis=1,inplace=True)
train.drop("commitdate",axis=1,inplace=True)

sample=True
if(sample==True):
    train=doSample(train)

print(train.shape)
print("*"*50)

#print(train.iloc[0,4])
col=["la","ld","lt","exp","rexp","sexp","ndev","pd","npt","entropy"]
temp=train[col]+1
#print(temp.iloc[0,1])
temp=np.log2(temp)
#print(temp.columns.tolist())

#print(temp.iloc[0,1])
#print(np.log2(1.006097561))
train[col]=temp
#print(train.iloc[0,4])
#print("*"*50)


print(train.corr())
print("*"*50)

print(train.var())
print(0==0.0)


