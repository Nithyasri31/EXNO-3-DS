## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```   
import pandas as pd

df=pd.read_csv("/content/Encoding Data (2).csv")

df
```
![image](https://github.com/user-attachments/assets/775acd23-3e10-4539-99b9-5adf3ac232fb)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/e0725f4e-b4ff-4303-b0a3-28cdc28800d9)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])

df
```
![image](https://github.com/user-attachments/assets/5a417a18-336a-4abb-83f1-7f679c973107)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/0700fbf8-ee9a-4376-848a-dcfbac53886e)
```
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2
```
![image](https://github.com/user-attachments/assets/ef472ae7-df16-48a4-b994-446fc2cf2ef7)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/268f6d88-c0ec-473b-91d2-92c0303d4ee1)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/66f4e30b-6dcf-4690-971c-eb97d015fb19)
```
from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data (2).csv")

df
```
![image](https://github.com/user-attachments/assets/a4e77519-b2b1-4ae9-85ec-265862ce1dbf)
```
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb
```
![image](https://github.com/user-attachments/assets/0c2aa5d7-7939-4849-9102-ea06466d3390)
```
from category_encoders import TargetEncoder

te=TargetEncoder

cc=df.copy()

te = TargetEncoder()
new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc
```
![image](https://github.com/user-attachments/assets/8ed6aa18-a3e8-43d1-a72d-145c0ebf40c0)
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform (1).csv")

df
```
![image](https://github.com/user-attachments/assets/153b371f-8934-46a4-9977-cb18acb5b706)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/4708ff04-e9f9-4a41-9ed5-aaf7e980e257)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4b6f752a-c148-4443-b484-c062881e1edf)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3766ce6f-62fa-4cd6-acbc-f47a36a7e1fc)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cf761ac3-50aa-4ecd-8d72-0488cfd6dd8f)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8d7b8ee6-de5b-4fc2-84e2-0716a034f71a)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/8279c75b-ede7-45da-8af0-2ba1acd7ab8d)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/ba8bb3ce-2def-42be-a30f-5c254e2355fa)
```
df["High Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/4d566724-bdaf-4509-bc16-e212add5b46e)
```
from  sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])

df
```
![image](https://github.com/user-attachments/assets/1cb94848-998b-4358-8fe6-bc0ad74f1db5)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a3471a3b-69dd-417c-be04-37022b0080fb)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/user-attachments/assets/32c3fa01-3979-4012-a1fc-edd222324c06)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c86ab840-1353-4133-84a9-f9dcd88c06dd)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/35a46bdb-6f00-419f-838d-0434a85071c7)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ac0dd6f3-075e-4580-9d80-9ae0947e3f2e)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt["Age"], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8f9cb825-bbc7-4099-956b-ab996f235a36)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/066c3829-1d3c-418a-ac9f-c8c2fb6725d2)

# RESULT:
       The given data and perform Feature Encoding and Transformation process and successsfuly saved the data to a file.

       
