import pandas as pd


a=pd.DataFrame({"a":[1,2,3],"b":[5,6,7]})

b=pd.DataFrame({"c":[11,22,33],"d":[55,66,77]})

print(pd.concat([b,a],axis=1))






