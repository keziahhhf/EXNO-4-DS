# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/e76ca0a2-9234-48ab-a377-d840f10b037d)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/9f2b4612-1233-4515-ad73-06ab2cb90969)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/07849754-9ac0-4d41-8259-d81981120ec9)

```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/0f3e3dd0-3320-4f5a-bc67-5e03d731f95e)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/ac82ceb4-b4f6-499d-be5c-da121144e3e1)
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/5a2c5d4e-3b58-4880-ab8e-4bdf582c239a)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/5ce6d5eb-54cb-4b9f-999f-bfe7a86b40ae)
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/d853c4d4-185e-4480-9ee3-f4b9da88c6d8)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/c3b8c2cf-185d-4970-98f5-2fd284435fa2)
```
data=data.dropna(axis=1)
data
```
![image](https://github.com/user-attachments/assets/12be4ff5-ebd5-42e8-b915-98e45fdb6724)
```
sal=data['SalStat']
data['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/a62b43dd-8e3e-40e6-af51-081afb656900)
```
sal2=data['SalStat']
new_data=pd.get_dummies(data,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/28b2702b-19f5-40a6-b709-4794982d4653)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/728e537b-518f-476b-8240-66ba3a32fbf5)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/0119c065-6905-457c-a477-19564d13eec7)
```

y=new_data['SalStat'].values
y
```
![image](https://github.com/user-attachments/assets/7adbeae6-49bf-4fbc-94b3-d2d4070bada5)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/bb2df54b-1b0a-4936-9c58-e0ad236f4e96)

#ALGORITHM IMPLEMENTATION
```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/1319a143-3515-4905-8e88-4a7e50fec241)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/89d08a7c-5dec-42fc-9891-6e2869c6b4b6)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/6463af71-b5ee-4004-8d22-37bce6631038)

```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/0302bae2-f07a-4c38-8ff5-62edf715bd87)
```
data.shape
```
![image](https://github.com/user-attachments/assets/8413e61a-4499-4d65-b88c-f0fd6c0ffc75)
```
import pandas as pd 
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
![image](https://github.com/user-attachments/assets/7a1571f5-c404-495a-853d-42d6bcaab762)
```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/97135884-8a87-45a0-a7e3-4a661a6bda6a)


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/17d7b0ec-da70-4b70-bbc0-50c73157c0a3)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/6847aca3-73e3-453d-8122-bb6f5269c13f)

# RESULT:
       Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
