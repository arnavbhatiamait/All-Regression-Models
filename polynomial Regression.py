# %% [markdown]
# Polynomial Regression
# 

# %% [markdown]
# importing The Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# impoting the Data set

# %%
df=pd.read_csv("Data.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# %% [markdown]
# Splting The Data Set Into Train and Test Set

# %%
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %% [markdown]
# Training The Polynomial Regression Model on The Training Data Set

# %%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lin_reg=LinearRegression()
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
lin_reg.fit(x_poly, y_train)

# %% [markdown]
# Predicting The Test Data Set

# %%
y_pred=lin_reg.predict(poly_reg.transform(x_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Evalution Of r2 score 

# %%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


