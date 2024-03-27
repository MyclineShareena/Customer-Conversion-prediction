#FinalProjectCustomerConversionPrediction

#Start with importing libraries wich we will use in the project

import pandas as pd
import numpy as np
import seaborn as sn           
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#%matplotlib inline

#For ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Import Data

data = pd.read_csv("Insurance company data.csv")
data.head()
data.columns
data.shape
data.dtypes

# Univariate Analysis

data['y'].value_counts()
 
# See the % distribution of the y
data['y'].value_counts(normalize=True)

# Bar plot of freequencies
data['y'].value_counts().plot.bar(color = np.random.rand(3,), ec='red')
sn.displot(data["age"], color=np.random.rand(3,))
data['job'].value_counts().plot.bar(color = np.random.rand(3,))
data['education_qual'].value_counts().plot.bar(color = np.random.rand(3,))

# Bivariate Analysis

print(pd.crosstab(data['job'],data['y']))
job=pd.crosstab(data['job'],data['y'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')

print(pd.crosstab(data['education_qual'],data['y']))
education_qual=pd.crosstab(data['education_qual'],data['y'])
education_qual.div(education_qual.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('education_qual')
plt.ylabel('Percentage')

print(pd.crosstab(data['marital'],data['y']))
marital=pd.crosstab(data['marital'],data['y'])
marital.div(marital.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('marital')
plt.ylabel('Percentage')

data['y'].replace('no', 0,inplace=True)
data['y'].replace('yes', 1,inplace=True)

corr = data.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
 
#missing values in the dataset.
data.isnull().sum()

# Model Building
target = data['y']
data = data.drop('y',1)
   
#Apply dummies on dataset
data = pd.get_dummies(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.25, random_state = 42)

# Logistic Regression
lreg = LogisticRegression()
# fiting the model on our data
lreg.fit(X_train,Y_train)
L_pred = lreg.predict(X_test)
print("Accuracy Score of Logistic Reg:",round(accuracy_score(Y_test, L_pred),2))
L_AUC_ROC = roc_auc_score(Y_test, L_pred)
print("ROC AUC Score of Logistic Reg:",round((L_AUC_ROC),2))

# Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=0, splitter='best')
dtc.fit(X_train, Y_train)
D_pred = dtc.predict(X_test)
print("Accuracy Score of Decision Tree Cl:",round(accuracy_score(Y_test, D_pred), 1))
D_AUC_ROC = roc_auc_score(Y_test, D_pred)
print("ROC AUC Score of Decision Tree Cl:",round(D_AUC_ROC, 2))

# K Neighbors Classifier
KNC = KNeighborsClassifier()
KNC.fit(X_train, Y_train)
K_pred = KNC.predict(X_test)
print("Accuracy Score of K Neighbors Cl:",round(accuracy_score(Y_test, K_pred), 2))
K_AUC_ROC = roc_auc_score(Y_test, K_pred)
print("ROC AUC Score of K Neighbors Cl:",round(K_AUC_ROC, 2))
