# %% [markdown]
# Multiple Linear Regression

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
y

# %% [markdown]
# Splitting Data into Train and Test set

# %%
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %% [markdown]
# Training The Multiple Linear Regression Model on the Data Set

# %%
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

# %% [markdown]
# Predicting Test results for the Multiple Linear Regression

# %%
y_pred=reg.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Evaluation of The test results for the Multiple Linear Regression

# %%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


