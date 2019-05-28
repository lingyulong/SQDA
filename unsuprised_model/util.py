import pandas as pd
import numpy as np
import random,math


data=pd.read_csv(r"E:\lihang\sqe\input\bugzilla.csv")
data=data[["lt","nf","la","ld"]]

print("*"*50)

#LA and LD was normalized by LT
def effortCal(data):
    lt=data["lt"]*data["nf"]
    lt[lt==0]=lt[lt==0]+1  #对于为0的数，设置为1
    effort=((data["la"]+data["ld"])*lt)/2
    return effort

def divideIntoKSubset(data,k=10):
    alist=[]
    nrow=data.shape[0]
    num=math.floor(nrow/k)
    rands=random.sample(range(nrow),nrow)
    #print(rands)
    for i in range(0,k):
        if i<k-1:
            #print(rands[i*num:(i+1)*num]) 
            alist.append(data.iloc[rands[i*num:(i+1)*num]])
        else:
            #print(rands[i*num:])
            alist.append(data.iloc[rands[i*num:]])
    return alist
'''
data=pd.DataFrame([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]])
alist=divideIntoKSubset(data,k=5)
print(alist[0])
print("*"*50)
print(alist[-1])
print("*"*50)

a=pd.DataFrame(columns=(0,1))
print(a)
print("*"*50)
print(pd.concat([a,alist[0]]))
print("*"*50)
for i in range(len(alist)):
    a=pd.concat([a,alist[i]])
print(a)

'''




'''    
#print(random.sample(range(data.shape[0]),data.shape[0]))
print(random.sample(range(3),3))
df=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
print(df)
print("*"*50)

df1=df.iloc[[0,2]]
print(df1)
print("*"*50)
df2=df.iloc[[0,1]]
print(df2)
print("*"*50)
alist=[]
alist.append(df1)
alist.append(df2)
print(alist)
print(type(alist))
print(alist[0])
print(type(alist[0]))
print("*"*50)
'''

'''
df2=df[[0,1]]
alist=[]
alist.append(df1)
alist.append(df2)
print(alist)
print(type(alist))
print(alist[0])
print(type(alist[0]))
print("*"*50)
df3=pd.DataFrame([list(df1),list(df2)])
print(df3)
'''
'''
df=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
print(df)
print("*"*50)

r=random.sample(range(df.shape[0]),df.shape[0])
print(r)
print("*"*50)
for i in range(math.ceil(df.shape[0]/2)):
    print(r[i:i*2])
print(r[0:2])
'''
'''
r=[1,2,3,4,5]
k=3
for i in range(0,len(r),k):
    print(r[i:i+k])
'''

def sortData(data,wst,bst):
    if(wst==False and bst==False):
        #print("mdl")
        #print(data.sort_values(by="per",ascending=False))
        #print("*"*50)
        return data.sort_values(by="per",ascending=False)
    if(wst==False and bst==True):
        #代码行越少越好，density越大越好
        #print("opt")
        #print(data.sort_values(by=["per","density","loc"],ascending=[False,False,True]))
        print("*"*50)
        return data.sort_values(by=["per","density","loc"],ascending=[False,False,True])
    if(wst==True and bst==False):
        #print("wst")
        #print(data.sort_values(by=["per","density","loc"],ascending=[False,True,False]))
        #print("*"*50)
        return data.sort_values(by=["per","density","loc"],ascending=[False,True,False])

#先计算第一个三角形的面积，在计算剩下的所有梯形的面积 
def computeArea(data):
    length=data.shape[0]
    
    cumX=np.cumsum(data["loc"])
    cumY=np.cumsum(data["num"])
    #print(data["num"])
    #print("*"*50)
    #print(cumY)
    #print(cumY[length-1])
    #print("*"*50)
    #print(cumX[length-1])
    x=cumX/cumX[length-1]
    y=cumY/cumY[length-1]
    #print(x)
    #print("*"*50)
    #print(y)
    #print("*"*50)
    areas=[]
    areas.append((x[0]*y[0]/2))  #计算第一个三角形的面积
    for i in range(1,length):    #计算剩下的所有梯形的面积 
        #print(x[i]-x[i-1])
        #print((y[i-1]+y[i]))
        #print("*"*50)
        area=(y[i-1]+y[i])*abs(x[i]-x[i-1])/2
        areas.append(area)
    #print(areas)
    #print(x[0])
    #print(x[1])
    #print(x[2])
   # print(np.sum(areas))
    return np.sum(areas)
    
    

def computePopt(data):

    #得到的data_mdl中的index是乱序的,所以用reset_indexUI重置索引，还会把index变成一列
    data_mdl=sortData(data,wst=False,bst=False).reset_index().drop("index",axis=1)
    data_opt=sortData(data,wst=False,bst=True).reset_index().drop("index",axis=1)
    data_wst=sortData(data,wst=True,bst=False).reset_index().drop("index",axis=1)

    mdl=computeArea(data_mdl)
    opt=computeArea(data_opt)
    wst=computeArea(data_wst)

    Popt=1-(opt-mdl)/(opt-wst)
    return Popt

def computeACC(data):
    
    length=data.shape[0]
    #计算ACC时，这里用的是mdl的排序，如果采用最坏的排序，那么得到的ACC很多都是0
    #因为，最坏的情况下，loc从大到小排序，这样很快loc就累积到了20%的代码量，这时的num还是特别少
    data_wst=sortData(data,wst=True,bst=False).reset_index().drop("index",axis=1)
    #print("data_wst")
   # print(data_wst.head(81))
    #print("*"*50)
    
    cumX=np.cumsum(data_wst["loc"])
    cumY=np.cumsum(data_wst["num"])
    
    x=cumX/cumX[length-1]
    #print("cumX")
    #print(cumX)
    #print("*"*50)
    #print("cumY")
    #print(cumY)
    #print("*"*50)
    #print("x")
    #print(x)
    #print("*"*50)
    pos=0
    for i in range(len(x)):
        if x[i]>=0.2:
            pos=i
            #print(pos)
            break
    #print(pos)
    ACC=cumY[pos]/cumY[length-1]
    #print(ACC)
    return ACC
 
'''
data=pd.DataFrame({"a":[1,3,1,4],"b":[8,7,6,5],"c":[10,9,12,11]})    
print(data)
print("*"*50)
sort_data=data.sort_values(by="a",ascending=True)
print(sort_data)
print("*"*50)
dd=sort_data.reset_index().drop("index",axis=1)
print(dd)
print("*"*50)
dd=dd.drop("index",axis=1)
print(dd)
print("*"*50)
#print(data.sort_values(by=["a","b"]))
#print("*"*50)
#print(data.sort_values(by=["a","b"],ascending=[False,False]))
#print("*"*50)
'''
#这里采用的采样是把0和1样本的数量保持一致，也就是如果0样本比1样本多，就把0样本减少到和1样本一样多
def doSample(data,seed=0):
    rs=np.random.RandomState(seed)
    
    dataN=data[data["bug"]==0]
    dataT=data[data["bug"]==1]
    numN=dataN.shape[0]
    numT=dataT.shape[0]
    num=numN
    #print(numN)
    #print(numT)
    #print("*"*50)
    if(numN>numT):  #如果非异常变更比异常变更多，就让他们一样多
        num=numT
        randomData=rs.randn(numN)
        randomData=pd.Series(randomData)
        randomData.sort_values(inplace=True)
        index=randomData.index
        index=index[:numT]
        dataN=dataN.iloc[index,:]
        
    else:
        randomData=rs.randn(numN)
        randomData=pd.Series(randomData)
        randomData.sort_values(inplace=True)
        index=randomData.index
        index=index[:numT]
        #print(index)
        #print("*"*50)
        dataT=dataT.iloc[index,:]
        #print(dataT)
        #print("*"*50)
    data=pd.concat([dataN,dataT])  #合并得到的df的索引是乱序的，所以重新给它个顺序的索引
    data.index=range(2*num)

    return data
    
'''
data=pd.DataFrame({"a":[1,2,3,4,5,6],"bug":[0,0,1,0,1,0]})
print(data)
print("*"*50)
a=doSample(data)
print(a)
print("*"*50)
'''
    
    


