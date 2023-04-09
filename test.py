import numpy as np
from scipy.signal import cheby1, freqz
import matplotlib.pyplot as plt
import pandas as pd

#filter
import seaborn as sns

dataset = pd.read_csv("S01G1AllRawChannels.csv")
frame = pd.DataFrame(dataset)

col = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
X = pd.DataFrame(frame, columns=col)

print(X.head(5))

print("before filter")

fig, ax = plt.subplots(figsize=(10,5))
column = col[1:]
for i in column:
    print(sns.lineplot(X[str(i)]))
    # X.index, 


plt.show()

