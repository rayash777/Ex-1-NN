
<H1>EX. NO.1</H1>
<H1> Introduction to Kaggle and Data preprocessing</H1>
<H5> Name  :RAYASH R</H5>
<H5>Reg no : 212224230226</H5>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>

STEP 2:Importing the dataset<BR>

STEP 3:Taking care of missing data<BR>

STEP 4:Encoding categorical data<BR>

STEP 5:Normalizing the data<BR>

STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```PY
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("/content/Churn_Modelling (1).csv")
data
data.head()
X=data.iloc[:,:-1].values
X
y=data.iloc[:,-1].values
y
data.isnull().sum()
data.duplicated()
data.describe()
data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train
X_test
print("Lenght of X_test ",len(X_test))
```
## OUTPUT:
## DATA SET:
<img width="785" height="352" alt="image" src="https://github.com/user-attachments/assets/aca0e3bb-f74b-4342-afcb-01bb0e877bf7" />


## X VALUES:
<img width="732" height="262" alt="image" src="https://github.com/user-attachments/assets/dc6d30ff-790d-423c-9b46-37408416fe03" />



## Y VALUES:
<img width="355" height="107" alt="image" src="https://github.com/user-attachments/assets/11b141be-033b-4d51-abc9-6aa296bd8472" />


## NULL VALUES:
<img width="262" height="698" alt="image" src="https://github.com/user-attachments/assets/44efb9ee-e90a-4be6-8361-d633d0ca173c" />



## DUPLICATED VALUES:
<img width="232" height="557" alt="image" src="https://github.com/user-attachments/assets/b4859804-64ba-4358-85a6-b6fc99df017a" />




## DESCRIPTION:
<img width="1003" height="432" alt="image" src="https://github.com/user-attachments/assets/74797f32-b0a8-442e-a346-df671eba47c7" />




## NORMALIES DATA SET:
<img width="1002" height="323" alt="image" src="https://github.com/user-attachments/assets/88dfe21f-3bc3-4219-a465-fb9e1af28dba" />




## TRAINING DATA:
<img width="770" height="702" alt="image" src="https://github.com/user-attachments/assets/d9577115-d495-414c-9a49-27b703a3da3f" />




## TESTING DATA:
<img width="810" height="546" alt="image" src="https://github.com/user-attachments/assets/56b3703b-5d26-4516-b929-6a3e1d82300f" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


