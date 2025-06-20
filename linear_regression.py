import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("score.csv")
print(data.info())
#the shape of the linear regression is Y=AX+B
#Ai:=Ai-alpha*(derivative of loss function over Ai)
X,Y=data["X"],data["Y"]
def grad(data,alpha,A,B):
    N=len(data)
    errA=0
    errB=0
    for i in range(N):
        errA+=((A*X[i]+B)-Y[i])*X[i]
        errB+=((A*X[i]+B)-Y[i])
    return A-2*alpha*errA/N,B-2*alpha*errB/N
def runGrad(data,alpha,iterations,A=0,B=0):
    for i in range(iterations):
        A,B=grad(data,alpha,A,B)
    return A,B
A,B=runGrad(data,0.001,1000)
plt.scatter(X,Y,label="point of Data")
x = np.linspace(0, 10, 100)
y = A * x + B

plt.plot(x, y, color='red', label="regression Model")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()
