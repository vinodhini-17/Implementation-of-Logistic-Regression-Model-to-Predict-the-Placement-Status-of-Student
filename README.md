# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Finally execute the program and display the output.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: vinodhini.k
RegisterNumber:  212223230245


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
*/
```
![image](https://github.com/user-attachments/assets/20f3a856-d3e4-4098-9b0e-47473b4cf7b0)

```
dataset.tail()
`````
![image](https://github.com/user-attachments/assets/cb8e845a-3d61-46d1-bfb4-3528131e4173)
````
dataset.info()
````
![image](https://github.com/user-attachments/assets/6acb10b2-2f8f-4e68-9af7-4c0f2db4e7fe)
````
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```````
![image](https://github.com/user-attachments/assets/464e0447-d385-45e6-9e99-9533d0a9016f)
````
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
`````
![image](https://github.com/user-attachments/assets/57700d39-2629-4858-bc2e-7bdb809c7eca)
````
dataset.info()
`````
![image](https://github.com/user-attachments/assets/9c8918d6-b9dd-4765-8623-8cf59af343f5)

````
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
````
![image](https://github.com/user-attachments/assets/1e53c0a1-a2d4-4ffe-85b1-54c05ee21b41)
````
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
````
![image](https://github.com/user-attachments/assets/abed60ad-2ba0-43ea-9858-311c2c71a447)
````
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
````
![image](https://github.com/user-attachments/assets/86a593d3-8871-4184-b8fb-ab6870abd7ec)
````
accuracy=accuracy_score(y_test, y_pred)
accuracy
````
![Uploading image.png…]()



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
