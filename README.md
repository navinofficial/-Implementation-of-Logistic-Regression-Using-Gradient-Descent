# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Navinkumar v
RegisterNumber:  212223230141
*/
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("placement.csv")  
data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 
data
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data 
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values
y 
theta = np.random.randn(x.shape[1]) 
Y=y 
def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,X,y,alpha,num_iterations): 
    m=len(y)
    for i in range(num_iterations): 
        h=sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h-y)/m 
        theta-=alpha * gradient 
        return theta
gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000) 
def predict(theta,X): 
    h=sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred 
y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy) 
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew)

```

## Output:
## Dataset :
![image](https://github.com/user-attachments/assets/426adb0b-918d-4ea3-9a37-0810fe4eb134)
## Information :
![image](https://github.com/user-attachments/assets/bbde5977-5d69-4abc-a78e-ddcfa15d46c9)
## Encoding:
![image](https://github.com/user-attachments/assets/e4809dc7-4869-4bb3-9b64-d3ecd98708a7)
## X and Y value:
![image](https://github.com/user-attachments/assets/53e635f1-6c66-4027-ade6-7a0e1b9c8faa)
![image](https://github.com/user-attachments/assets/d901bda1-c361-4df5-87eb-7faf1940da64)
## Gradient Descent:
![image](https://github.com/user-attachments/assets/a73d07f2-a4c8-47a5-92bf-e45e0f0fcb8b)
## Accuracy:
![image](https://github.com/user-attachments/assets/4ddec6eb-87b6-4a67-992b-ffefe2701ff2)
## Prediction:
![image](https://github.com/user-attachments/assets/a8c1a7b9-d599-4d47-8e71-7bdcf82bb787)
![image](https://github.com/user-attachments/assets/002f95a8-fda0-4a2a-829d-a0b95d9d4d2f)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

