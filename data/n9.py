import random
import numpy as np

inputs =[]
asnwers =[]
for i in range(2000):
    inputs.append([round(random.uniform(0, 1),5), round(random.uniform(0, 1),5), round(random.uniform(0, 1),5) ])
    asnwers.append(inputs[i][0]*inputs[i][1]*inputs[i][2])

inputs = np.array(inputs)
asnwers = np.array(asnwers).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=12, activation='relu', input_dim=3))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile('adam', 'mean_absolute_error')
model.fit(inputs, asnwers, epochs=350)

for i in range(10):
    predict = np.array([[round(random.uniform(0,1), 2),round(random.uniform(0,1), 2),round(random.uniform(0,1), 2)]])
    answer = model.predict(predict)
    print("Результат произведения чисел ",predict, " = ", round(predict[0][0]*predict[0][1]*predict[0][2],2),"(В ручную), ", round(answer[0][0],2), "(Нейронная сеть)")