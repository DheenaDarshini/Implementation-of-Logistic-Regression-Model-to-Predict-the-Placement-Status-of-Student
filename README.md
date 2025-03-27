# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection & Preprocessing

2. Select relevant features that impact placement

3. Import the Logistic Regression model from sklearn.

4. Train the model using the training dataset.

5. Use the trained model to predict placement for new student data.


## Program:
```
Developed by: Dheena Darshini Karthik Dheepan
RegisterNumber: 212223240030

```
~~~
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
~~~

## Output:
![image](https://github.com/user-attachments/assets/e309e40a-c6e2-444b-95dd-54d723ee762d)

~~~
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
~~~

## Output:
![image](https://github.com/user-attachments/assets/042561aa-79bf-456f-aad6-ff51c432ad25)

~~~
data1.isnull().sum()
~~~

## Output:
![image](https://github.com/user-attachments/assets/3155b658-4d54-4540-bd83-d60552752490)

~~~
data1.duplicated().sum()
~~~

## Output:
![image](https://github.com/user-attachments/assets/45fff0eb-77f1-484e-b4fc-8e0caac4d082)


~~~
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
~~~

## Output:
![image](https://github.com/user-attachments/assets/60d2b836-d6da-4ab8-8d93-14a616946ef0)

~~~
x=data1.iloc[:,:-1]
x
~~~

## Output:
![image](https://github.com/user-attachments/assets/18acff13-4d99-405b-8f0a-683d314e3435)

~~~
y=data1["status"]
y
~~~

## Output:
![image](https://github.com/user-attachments/assets/5fbb9446-c230-47b4-8903-4869129625dc)

~~~
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
~~~
~~~
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
~~~

## Output:
![image](https://github.com/user-attachments/assets/6184688e-0a52-4b04-bf93-fc3214efb1fa)


~~~

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
~~~

## Output:
![image](https://github.com/user-attachments/assets/ebc3f376-e8d6-4932-a654-eefbeb4575e7)

~~~
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
~~~
## Output:
![image](https://github.com/user-attachments/assets/e9276cac-161e-4194-ab52-7e0b139043f3)

~~~

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

## Output:
![image](https://github.com/user-attachments/assets/85458802-fcaa-47b5-9beb-b252cb86fcee)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
