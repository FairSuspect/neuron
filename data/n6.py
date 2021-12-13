import random
import numpy as np

inputs =[]
results =[]
for i in range(1000):
    inputs.append([random.uniform(1, 1000), random.uniform(1, 1000)])
    results.append(sum(inputs[i]))

inputs = np.array(inputs)
results = np.array(results).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=6, activation='relu', input_dim=2))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile('adam', 'mean_squared_error')
model.fit(inputs, results, epochs=200)

predict = np.array([[3,1], [2, 9], [1,1], [0,0], [0,3]])
answer = model.predict(predict)
print("Значение суммы ",predict,": ", answer)
