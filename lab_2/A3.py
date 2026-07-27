import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def my_mean(data):
    s=0
    for i in data:
        s+=i
    return s/len(data)

def my_variance(data):
    m=my_mean(data)
    s=0
    for i in data:
        s+=(i-m)**2
    return s/len(data)
df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="IRCTC Stock Price")
price=df["Price"]
chg=df["Chg%"]
print("Mean using NumPy:",np.mean(price))
print("Variance using NumPy:",np.var(price))
print("Mean using Function:",my_mean(price))
print("Variance using Function:",my_variance(price))
numpy_time=[]
own_time=[]
for i in range(10):
    start=time.time()
    np.mean(price)
    np.var(price)
    numpy_time.append(time.time()-start)
    start=time.time()
    my_mean(price)
    my_variance(price)
    own_time.append(time.time()-start)
print("Average NumPy Time:",np.mean(numpy_time))
print("Average Own Function Time:",np.mean(own_time))
wed=df[df["Day"]=="Wed"]
print("Wednesday Mean:",wed["Price"].mean())
apr=df[df["Month"]=="Apr"]
print("April Mean:",apr["Price"].mean())
loss=len(df[df["Chg%"]<0])
print("Probability of Loss:",loss/len(df))
wed_profit=wed[wed["Chg%"]>0]
print("Probability of Profit on Wednesday:",len(wed_profit)/len(df))
if len(wed)>0:
    print("Conditional Probability:",len(wed_profit)/len(wed))
else:
    print("Conditional Probability: 0")
plt.scatter(df["Day"],df["Chg%"])
plt.xlabel("Day")
plt.ylabel("Chg%")
plt.title("Day vs Chg%")
plt.show()