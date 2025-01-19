#SOM

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
ms = MiniSom(x=10, y=10, input_len=15,learning_rate=0.5,sigma =1)
ms.random_weights_init(X)
ms.train_random(data=X, num_iteration=100)

from pylab import pcolor, bone, colorbar, plot, show
bone()
pcolor(ms.distance_map().T)
markers = ['o','s']
colors = ['r','g']
colorbar()
for i,x in enumerate(X):
    w = ms.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mapping = ms.win_map(X)
frauds = np.concatenate( (mapping[(6,3)], mapping[(1,3)]),axis = 0)
frauds = sc.inverse_transform(frauds)
