import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

cols=["ns","nm","nf","entropy","lt","fix","ndev","pd","nuc","exp","rexp","sexp"]

us_Popt=pd.read_csv(r"./output/unsuprised/us_Popt.csv")
us_ACC=pd.read_csv(r"./output/unsuprised/us_ACC.csv")
us_Popt=us_Popt[cols]
us_ACC=us_ACC[cols]


s_Popt=pd.read_csv(r"./output/suprised/s_Popt.csv")
s_ACC=pd.read_csv(r"./output/suprised/s_ACC.csv")
#s_Popt=s_Popt[cols]
#s_ACC=s_ACC[cols]

Popt=pd.concat([s_Popt,us_Popt],axis=1)
ACC=pd.concat([s_ACC,us_ACC],axis=1)



plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(10,3))
Popt.boxplot()
plt.xlabel("Popt")
plt.title("十折交叉验证下的有监督和无监督模型的Popt")
plt.show()

plt.figure(figsize=(10,3))
ACC.boxplot()
plt.xlabel("ACC")
plt.title("十折交叉验证下的有监督和无监督模型的ACC")
plt.show()
    

