# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the dataset and required libraries (pandas, sklearn, matplotlib).
2. Encode the categorical column (salary) using LabelEncoder.
3. Split the dataset into features (x) and target (y), then apply train_test_split.
4. Train a DecisionTreeClassifier using the training data and make predictions.
5. Evaluate the model's accuracy and visualize the decision tree using plot_tree.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rohit GP
RegisterNumber:  212224220082
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
print("Null values:\n",data.isnull().sum())
print("Class distribution:\n",data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

print(x.head())
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

from sklearn import metrics
y_pred =  dt.predict(x_test)

accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

sp=dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample_predicts:",sp)
```

## Output:

![image](https://github.com/user-attachments/assets/b4ba4f51-589b-4d4f-b1e7-2b755c31a7cd)

![image](https://github.com/user-attachments/assets/1040dfd8-bcd2-4310-b2d7-02ac9387f7d4)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
