# %% [markdown]
# Support vector regression

# %% [markdown]
# Importing The Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# importing the dataset

# %%
df=pd.read_csv("Data.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y=y.reshape(len(y),1)

# %% [markdown]
# Splitting Data into Train and Test set

# %%
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
y_train=sc_y.fit_transform(y_train)

# %% [markdown]
# Training The SVR Model

# %%
from sklearn.svm import SVR
reg=SVR(kernel="rbf")
reg.fit(x_train,y_train)

# %% [markdown]
# Predicting The Test Set Results

# %%
y_pred = sc_y.inverse_transform(reg.predict(sc_x.transform(x_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# R2 score for svr model

# %%
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


