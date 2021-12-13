import random
import numpy as np

X =[]
y =[]
for i in range(1000):
    X.append([random.randint(1, 1000), random.randint(1, 1000)])
    y.append(sum(X[i]))

X = np.array(X)
y = np.array(y).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=6, activation='relu', input_dim=2))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile('adam', 'mean_squared_error')
model.fit(X, y, epochs=200)

pred = np.array([[3,1], [2, 9], [1,1], [0,0], [0,3]])
#pred = sclr.transform(pred)
answer = model.predict(pred)
#answer = sclr2.inverse_transform(answer)
d = dict(pred,answer)
print("Значение суммы ",pred,": ", answer)
print(d)