import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
cols=["ns","nm","nf","entropy","lt","fix","ndev","pd","nuc","exp","rexp","sexp"]


Popt=pd.read_csv(r"../output/unsuprised/us_Popt.csv")
ACC=pd.read_csv(r"../output/unsuprised/us_ACC.csv")
Popt=Popt[cols]
ACC=ACC[cols]
print(Popt)
print("*"*50)
print(ACC)
print("*"*50)

plt.figure(figsize=(6,2))
Popt.boxplot()
plt.show()
plt.figure(figsize=(6,2))
ACC.boxplot()
plt.show()
    

