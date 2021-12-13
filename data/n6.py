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

model.add(Dense(output_dim=6, activation='relu', input_dim=2))
model.add(Dense(output_dim=12, activation='relu'))
model.add(Dense(output_dim=12, activation='relu'))
model.add(Dense(output_dim=6, activation='relu'))
model.add(Dense(output_dim=1, activation='linear'))

model.compile('adam', 'mean_squared_error')
model.fit(X, y, epochs=200)

pred = np.array([[7,65]])
#pred = sclr.transform(pred)
predd = model.predict(pred)
#predd = sclr2.inverse_transform(predd)

print("Значение суммы ",pred,": ", predd)