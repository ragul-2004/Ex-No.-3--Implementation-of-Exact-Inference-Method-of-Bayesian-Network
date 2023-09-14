# Ex No. 3- Implementation of Exact Inference Method of Bayesian Network

## Aim:
To implement the inference Burglary P(B| j,⥗m) in alarm problem by using Variable Elimination method in Python.

## Algorithm:

Step 1: Define the Bayesian Network structure for alarm problem with 5 random 
             variables, Burglary,Earthquake,John Call,Mary Call and Alarm.<br>
Step 2: Define the Conditional Probability Distributions (CPDs) for each variable 
            using the TabularCPD class from the pgmpy library.<br>
Step 3: Add the CPDs to the network.<br>
Step 4: Initialize the inference engine using the VariableElimination class from 
             the pgmpy library.<br>
Step 5: Define the evidence (observed variables) and query variables.<br>
Step 6: Perform exact inference using the defined evidence and query variables.<br>
Step 7: Print the results.<br>

## Program :
```
NAME:RAGUL A C
REG NO:212221240042
```
```
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
class BayesClassifier:
  def __init__(self):
    self.clf = GaussianNB()
  def fit(self, X, y):
    self.clf.fit(X, y)
  def predict(self, X):
    return self.clf.predict(X)
iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = 38)
clf = BayesClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy: ",accuracy)
```


## Output :



![image](https://github.com/ragul-2004/Ex-No.-3--Implementation-of-Exact-Inference-Method-of-Bayesian-Network/assets/94367917/bfc265f0-3a5d-4f4f-9d53-698cb3931e14)





## Result :  

Hence, Bayes classifier for iris dataset is implemented successfully
