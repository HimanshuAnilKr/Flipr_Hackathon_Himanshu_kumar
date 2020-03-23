import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#Importing the dataset 
df_train = pd.read_csv("Train_dataset.csv")
df_test = pd.read_csv("Test_dataset.csv")
#df_test = pd.read_csv("Test_dataset.csv")
#Extracting feature from df_train
y=df_train.iloc[:,27]
df_train=df_train.iloc[:,0:27]
#concatinating both test and train data_set
df=pd.concat([df_train,df_test],axis=0)
#To apply various operations on the test+train dataset 
X=df
#Deciding continuous and categorical variable
X_cat=X.iloc[:,[1,2,3,5,6,7,8,9,10,11,14,15]]
X_cont=X.iloc[:,[4,12,13,16,17,18,19,20,21,22,24,25]]
#assigning vacant values to null

X_cont=pd.DataFrame(X_cont)
X_cat=pd.DataFrame(X_cat)

X_cont=X_cont.values
X_cat=X_cat.values

X_cont[pd.isnull(X_cont)]='NaN'
X_cat[pd.isnull(X_cat)]='NaN'

for i in range(0,25212):
    X_cont[i][0]=len(X_cont[i][0])

#Applying label encoding to the categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler
label_x=LabelEncoder()
X_cat[:,0]=label_x.fit_transform(X_cat[:,0])
X_cat[:,1]=label_x.fit_transform(X_cat[:,1])
X_cat[:,2]=label_x.fit_transform(X_cat[:,2])
X_cat[:,3]=label_x.fit_transform(X_cat[:,3])
X_cat[:,5]=label_x.fit_transform(X_cat[:,5])
X_cat[:,9]=label_x.fit_transform(X_cat[:,9])
X_cat[:,10]=label_x.fit_transform(X_cat[:,10])

#Applying label encoding to the continuous variable
#Dealing with missing values of continuous variables 
X_cat=X_cat.reshape(25212,12,1)

imp=Imputer(missing_values='NaN',axis=0,strategy='most_frequent')
X_cat[:,4]=imp.fit_transform(X_cat[:,4])


X_cat=X_cat.reshape(25212,12)
X_cat[:,6]=label_x.fit_transform(X_cat[:,6])
X_cat=X_cat.reshape(25212,12,1)
imp=Imputer(missing_values=1,axis=0,strategy='most_frequent')
X_cat[:,6]=imp.fit_transform(X_cat[:,6])
X_cat=X_cat.reshape(25212,12)
X_cat[:,6]=label_x.fit_transform(X_cat[:,6])

X_cat=X_cat.reshape(25212,12)
X_cat[:,11]=label_x.fit_transform(X_cat[:,11])
X_cat=X_cat.reshape(25212,12,1)
imp=Imputer(missing_values=1,axis=0,strategy='most_frequent')
X_cat[:,11]=imp.fit_transform(X_cat[:,11])
X_cat=X_cat.reshape(25212,12)
X_cat[:,11]=label_x.fit_transform(X_cat[:,11])

imp=Imputer(missing_values='NaN',axis=0,strategy='mean')
X_cont=imp.fit_transform(X_cont)


#Now concatenating the categorical and continuous variable
y=np.asarray(y)
X_cat=X_cat.reshape(25212,12)
x=np.zeros((25212,24))
x[:,0:12]=X_cat
x[:,12:24]=X_cont
'''
from sklearn.utils import shuffle
x,y=shuffle(x,y,random_state=132)

#Splitting the training set into test set and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

y_train=y_train.reshape(9642,1)
y_test=y_test.reshape(1072,1)
'''
#Splitting test and train dataset 
X_train=x[0:10714,:]
X_test=x[10714:,:]

#Applying Random Forest Regressor instead of classifier
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=270)
reg.fit(X_train,y)

#Printing values for test dataset 
y_pred=reg.predict(X_test)
'''
from sklearn.metrics import mean_squared_error
loss=mean_squared_error(y_test,y_pred)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
'''
y_pred=y_pred.reshape(14498,1)
#Converting into csv file
ss=pd.DataFrame(y_pred)
ss.to_csv("corona_result1.csv",header="Infect_Prob", index = False)










