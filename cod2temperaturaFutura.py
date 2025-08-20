import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# criar dados ficticios
np.random.seed(42)
n = 200;

hora = np.random.randint(0,24,n)
umidade = np.random.randint(0,90,n)
vento = np.random.randint(0,40,n)

temperatura  = (15 + 0.5*hora - 0.2*umidade -0.1*vento + np.random.normal(0,2,n));

df= pd.DataFrame({'hora':hora,'umidade':umidade,'vento':vento,'temperatura':temperatura})

print("primeiro  dados do conjunto")
print(df.head())

X = df[["hora","umidade","vento"]]
y = df["temperatura"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

modelo = LinearRegression()
modelo.fit(X_train,y_train)

y_pred = modelo.predict(X_test)

print("coeficient", modelo.coef_)
print("inter",modelo.intercept_)
print("Erro quadradico:(MSE)",mean_squared_error(y_test,y_pred))
print("R-squared:",r2_score(y_test,y_pred))

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,color="blue",alpha=0.7)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--",lw=2)

plt.xlabel("temperatura real")
plt.ylabel("temperatura prevista")
plt.title("Comparação entre temperatura real e prevista")
plt.grid(True)
plt.show()