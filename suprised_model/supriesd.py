import pandas as pd
import numpy as np
from util import effortCal,computePopt,computeACC,doSample
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")

'''
train=pd.read_csv(r"E:\lihang\sqe2\input\bugzilla.csv")
test=train.copy()

train.drop("transactionid",axis=1,inplace=True)
train.drop("commitdate",axis=1,inplace=True)
test.drop("transactionid",axis=1,inplace=True)
test.drop("commitdate",axis=1,inplace=True)

print(train.dtypes)
print("*"*50)
'''
def suprised_Popt_ACC(train,test):

    print("^"*50)
    print(train.shape)
    print(train[train["bug"]==0].shape[0])
    print(train[train["bug"]==1].shape[0])
    print(test.shape)
    print(test[test["bug"]==0].shape[0])
    print(test[test["bug"]==1].shape[0])
    print("*"*50)
    
    sample=True
    if(sample==True):
        train=doSample(train)
    print(train.shape)
    print(train[train["bug"]==0].shape[0])
    print(train[train["bug"]==1].shape[0])
    print(test.shape)
    print(test[test["bug"]==0].shape[0])
    print(test[test["bug"]==1].shape[0])
    print("*"*50)
    
    
    
    #这一步只是产生了一列有效值,意思就是代码行loc
    train_churn=effortCal(train)
    test_churn=effortCal(test)
     
    
    #计算bug密度，加1是为了防止有效值中·有0
    train_bugdensity=train["bug"]/(train_churn+1)
    test_bugdensity=test["bug"]/(test_churn+1)
    
    train_bug=train["bug"]
    test_bug=test["bug"]
    
    
    
    
    #log变换
    col=["la","ld","lt","exp","rexp","sexp","ndev","pd","npt","entropy"]
    
    temp=np.log2(train[col]+1)
    train[col]=temp
    
    temp=np.log2(test[col]+1)
    test[col]=temp
    
    
    #rexp与exp的相关系数是0.928
    #nm与ns的相关系数是0.87
    train=train.drop(["nm","rexp"],axis=1)
    print(train.shape)
    
    #删除没有信息量的变量，也就是var为0的变量
    train_std=train.std()
    print(train_std)
    del_col=train_std[train_std==0].index.tolist()
    train=train.drop(del_col,axis=1)
    print(train.shape)
    print(train.columns.tolist())
    print("*"*50)
    
    
    #删除la与ld
    train=train.drop(["la","ld"],axis=1)
    print(train.shape)
    print(train.columns.tolist())
    print("*"*50)
    
    #['ns', 'nf', 'entropy', 'lt', 'fix', 'ndev', 'pd', 'npt', 'exp', 
    # 'sexp', 'bug']
    
    
    def test_build_model(train,test,target="bug"):
        col=train.columns.tolist()
        col.remove(target)
        
        
        model=GaussianNB()
        model.fit(train[col],train[target])
     
        y_train=model.predict_proba(train[col])[:,1]
        y_test=model.predict_proba(test[col])[:,1]
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        #print(y_train)
        #print(y_test)
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("LogisticRegression")
        model=LogisticRegression(penalty='l2')
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("KNeighborsClassifier")
        model=KNeighborsClassifier()
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("tree")
        model=tree.DecisionTreeClassifier()
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        y_train=[round(i,5) for i in y_train]
        y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("SVC")
        model=SVC(kernel="rbf",probability=True)
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        print("RandomForestClassifier")
        model=RandomForestClassifier(n_estimators=8)
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("GradientBoostingClassifier")
        model=GradientBoostingClassifier()
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        y_train=[round(i,5) for i in y_train]
        y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        print("AdaBoostClassifier")
        model=AdaBoostClassifier()
        model.fit(train[col],train[target])
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        
        #y_train=[round(i,5) for i in y_train]
        #y_test=[round(i,5) for i in y_test]
        
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)    
        return y_train,y_test
    
    def build_model(train,test,ori_model,target="bug"):
        col=train.columns.tolist()
        col.remove(target)
    
        model=ori_model
        model.fit(train[col],train[target])
     
        y_train=model.predict_proba(train[col])[:,1].tolist()
        y_test=model.predict_proba(test[col])[:,1].tolist()
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        y_train.sort()
        y_test.sort()
        y_train=[round(i,5) for i in y_train]
        y_test=[round(i,5) for i in y_test]
        
        #print(y_train)
        #print(y_test)
        print(len(y_train)-len(np.unique(y_train)))
        print(len(y_test)-len(np.unique(y_test)))
        print("*"*50)
        
        return y_train,y_test
    
    def Popt_ACC(y_train,y_test):
        y_train=pd.DataFrame({"per":y_train,"num":train_bug.values,"density":train_bugdensity.values,"loc":train_churn})
        y_test=pd.DataFrame({"per":y_test,"num":test_bug.values,"density":test_bugdensity.values,"loc":test_churn})
    
        #print(y_train) 
        #train_Popt=computePopt(y_train)
        #train_ACC=computeACC(y_train)
        
        test_Popt=computePopt(y_test)
        test_ACC=computeACC(y_test)
        #test_Popt=0
        #test_ACC=0
        return test_Popt,test_ACC
    
    nb=GaussianNB()
    #lr=LogisticRegression(penalty='l2')
    knn=KNeighborsClassifier()
    dt=tree.DecisionTreeClassifier()
    svc=SVC(kernel="rbf",probability=True)
    rfc=RandomForestClassifier(n_estimators=8)
    gbtc=GradientBoostingClassifier() 
    adtc=AdaBoostClassifier()
    models=[nb,knn,dt,svc,rfc,gbtc,adtc]
    models_name=["nb","knn","dt","svc","rfc","gbtc","adtc"]
    test_Popt={}
    test_ACC={}
    
    i=0
    for model in models:
        y_train,y_test=build_model(train,test,model)
    
        temp_Popt,temp_ACC=Popt_ACC(y_train,y_test)
        print(test_Popt)
        print(test_ACC)
        test_Popt[models_name[i]]=temp_Popt
        test_ACC[models_name[i]]=temp_ACC
        i=i+1
        print("^"*50)
        
    print(test_Popt)
    print()
    print(test_ACC)
    return test_Popt,test_ACC

