import random
import numpy as np
import math

X =[]
y =[]

for i in range(1000):
    X.append(random.randint(0, 360))
    y.append(math.radians(X[i]))

X = np.array(X)
y = np.array(y).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=6, activation='relu', input_dim=1))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile('adam', 'mean_squared_error')
model.fit(X, y, epochs=100)

pred = np.array([180, -180, 360 , -0, 90, -90, 45])
answer = model.predict(pred)

print(pred, " градусов в радианы: \n", answer)
