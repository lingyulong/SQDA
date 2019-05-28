import pandas as pd
import numpy as np
from util import effortCal,computePopt,computeACC

'''
train=pd.read_csv(r"E:\lihang\sqe\input\bugzilla.csv")
train.drop("transactionid",axis=1,inplace=True)
train.drop("commitdate",axis=1,inplace=True)
#print(data.head(1))
#print("*"*50)
#print(data.describe())
#print("*"*50)
'''

def unsuprised_Popt_ACC(train,test):
    #这一步只是产生了一列有效值,意思就是代码行loc
    train_churn=effortCal(train)
    test_churn=effortCal(test)
    
    
    #计算bug密度，加1是为了防止有效值中·有0
    train_bugdensity=train["bug"]/(train_churn+1)
    test_bugdensity=test["bug"]/(test_churn+1)
    
    train_bug=train["bug"]
    test_bug=test["bug"]
    
    train["lt"]=train["lt"]*train["nf"]
    train["la"]=train["la"]*train["lt"]
    train["ld"]=train["ld"]*train["lt"]
    train["nuc"]=train["npt"]*train["nf"]
    
    test["lt"]=test["lt"]*test["nf"]
    test["la"]=test["la"]*test["lt"]
    test["ld"]=test["ld"]*test["lt"]
    test["nuc"]=test["npt"]*test["nf"]
    
    #前面几不一共：train:改变了lt、la、ld、添加了一列bugdensity
    
    def build_model(train,test,name):
        y_train=10000/(train[name]+1)
        y_test=10000/(test[name]+1)
        return y_train,y_test
    
    
    def Popt_ACC(y_train,y_test):
        y_train=pd.DataFrame({"per":y_train,"num":train_bug.values,"density":train_bugdensity.values,"loc":train_churn})
        y_test=pd.DataFrame({"per":y_test,"num":test_bug.values,"density":test_bugdensity.values,"loc":test_churn})
    
        #print(y_train) 
        #train_Popt=computePopt(y_train)
        #train_ACC=computeACC(y_train)
        
        test_Popt=computePopt(y_test)
        test_ACC=computeACC(y_test)
        
        return test_Popt,test_ACC
    
    
    labels=["ns","nm","nf","entropy","lt","fix","ndev","pd","nuc","exp","rexp","sexp"]
    Popt={}
    ACC={}
    
    
    for label in labels:
        y_train,y_test=build_model(train,test,label)
        #print("bug")
        #print(train_bug)
        #print("*"*50)
        #print("loc")
        #print(train_churn)
        #print("*"*50)
        #print("prefict")
        #print(y_train)
        #print("*"*50)
        
        test_Popt,test_ACC_temp=Popt_ACC(y_train,y_test)
        Popt[label]=test_Popt
        ACC[label]=test_ACC_temp
    
    #print(Popt)
    print(ACC)
    return Popt,ACC


    
    










