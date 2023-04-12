from sre_parse import fix_flags
import numpy as np
from scipy.signal import cheby1, freqz, butter, lfilter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

#filter
import seaborn as sns

#cnn
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import BatchNormalization, LeakyReLU



dataset = pd.read_csv("S01G1AllRawChannels.csv")
frame = pd.DataFrame(dataset)

col = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
X = pd.DataFrame(frame, columns = col)

# print(X.head(5))
X.head(5)

print("before filter")

fig, ax = plt.subplots(figsize=(10,5))
column = col[1:]
for i in column:
    sns.lineplot(X[str(i)])
    # print(sns.lineplot(X[str(i)]))

# to display the graph 
# plt.show()

#chebyshev
measurements = 2880
time = 10
sampling_rate = measurements / time
print(sampling_rate)

nyquist = sampling_rate * 0.5

lowcut = 8
highcut = 12
low = lowcut / nyquist
high = highcut / nyquist
b, a = cheby1(3, 4, [low, high], btype= 'bandpass') 
final = {}

for i in column:
    filtered_sig = lfilter(b, a, X[i].values)
    intermediate_dictionary = {i:filtered_sig}
    final.update(intermediate_dictionary)

filtered_dataframe = pd.DataFrame(final)

#after filter
fig, ax = plt.subplots(figsize=(10,5))

for i in column:
    sns.lineplot(filtered_dataframe[i])

#plt.show()

#cnn
from sklearn.model_selection import train_test_split

X = dataset.drop('Outcome', axis=1)
Y = dataset['Outcome']=='HAPV'
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

X_train1 = X_train.values.reshape(28689,14,1)
X_test1 = X_test.values.reshape(9563,14,1)

X_train1.shape
#print

X_test1.shape
#print

epochs = 20
num_classes = 1


model = Sequential()
model.add(Conv1D(32, (1), input_shape=(14,1), activation = 'relu'))
model.add(MaxPooling1D((2), padding = 'same'))
model.add(Dropout(0.25))
model.add(Conv1D(64, (3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling1D(pool_size = (2), padding = 'same'))
model.add(Dropout(0.25))
model.add(Conv1D(128, (3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling1D(pool_size = (2), padding = 'same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()



