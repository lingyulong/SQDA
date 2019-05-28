import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

Popt=pd.read_csv(r"../output/suprised/s_Popt.csv")
ACC=pd.read_csv(r"../output/suprised/s_ACC.csv")


plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(10,3))
Popt.boxplot()
plt.xlabel("Popt")
plt.show()

plt.figure(figsize=(10,3))
ACC.boxplot()
plt.xlabel("ACC")
plt.show()
    

