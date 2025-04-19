# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries 
2. Load the Dataset 
3. Explore and Preprocess the Data 
4. Split the Dataset
5. Train the Decision Tree Regressor
6. Make Predictions
7. Evaluate the Model
8. Visualize the Results 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Divya R V 
RegisterNumber:212223100005  
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2

mse= metrics.mean_squared_error(y_test,y_pred)
mse
dt.predict([[5,6]])

```


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![Screenshot 2025-04-15 091840](https://github.com/user-attachments/assets/f2360c8b-da59-4dfd-b053-d78983740a60)

![Screenshot 2025-04-15 091847](https://github.com/user-attachments/assets/e6bd5315-7281-4b1b-b454-f35f8cecf8b7)

![Screenshot 2025-04-15 091853](https://github.com/user-attachments/assets/47921b80-7f3e-4c88-80be-77dd92d06781)

![Screenshot 2025-04-15 091912](https://github.com/user-attachments/assets/4e444b57-0877-443f-9a0d-8dbd9a41e4b8)

![Screenshot 2025-04-15 091918](https://github.com/user-attachments/assets/b91bb806-947a-4f87-8fd3-ba38cece2f3c)

![Screenshot 2025-04-15 091925](https://github.com/user-attachments/assets/646a2e21-34f2-4757-8975-7ccd188bf5e1)

![Screenshot 2025-04-15 091932](https://github.com/user-attachments/assets/f95b3214-b42c-4226-93a2-ffd70415b0f6)

![Screenshot 2025-04-15 091938](https://github.com/user-attachments/assets/8330795b-50f5-450e-aa00-2382e0705972)

![Screenshot 2025-04-15 091944](https://github.com/user-attachments/assets/78fca40b-4a31-46a9-bb51-7e9f06c77930)

![Screenshot 2025-04-15 091953](https://github.com/user-attachments/assets/e7c06e01-3e39-4a9c-bec0-44c645d67d70)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
