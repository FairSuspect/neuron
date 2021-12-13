import random
import numpy as np

X =[]
y =[]
for i in range(2000):
    X.append([round(random.uniform(0, 1),5), round(random.uniform(0, 1),5), round(random.uniform(0, 1),5) ])
    y.append(X[i][0]*X[i][1]*X[i][2])

X = np.array(X)
y = np.array(y).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=12, activation='relu', input_dim=3))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile('adam', 'mean_absolute_error')
model.fit(X, y, epochs=300)

for i in range(10):
    pred = np.array([[round(random.uniform(0,1), 2),round(random.uniform(0,1), 2),round(random.uniform(0,1), 2)]])
    predd = model.predict(pred)
    print("Значение произведения чисел ",pred, " = ", round(pred[0][0]*pred[0][1]*pred[0][2],2),": ", round(predd[0][0],2))