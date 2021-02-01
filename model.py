import pandas as pd
import numpy as np

df = pd.read_csv("Salary (2).csv")

x = df[["YearsExperience"]]
y = df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)

x = model.predict([[5]])
print(x)

import pickle
pickle.dump(model,open("model.pkl","wb"))