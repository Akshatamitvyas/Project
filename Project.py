import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df1=pd.read_csv(r'C:\\Users\\akshat vyas\\Desktop\\Sixth Sem\\AsknBid\\Project\\fundamentals.csv')
df2=pd.read_csv(r'C:\\Users\\akshat vyas\\Desktop\\Sixth Sem\\AsknBid\\Project\\prices-split-adjusted.csv')
df3=pd.read_csv(r'C:\\Users\\akshat vyas\\Desktop\\Sixth Sem\\AsknBid\\Project\\securities.csv')
d1,d2,d3=df1,df2,df3

f=[[:,0,1,4,5]]
l=[2,3]
LR=LinearRegression()
t=LR.fit(f,l)
r=t.predict([[6]])
print(r)
