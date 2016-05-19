import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

f = open('results_tensor2.csv','r')
XY_data=[]
X_data = []
y_data = []
for line in f:
    data = line.strip().split(',')
    XY_data.append(data)
    X_data.append(data[:56])
    y_data.append(data[56:])
X_train = X_data[:5000]
y_train = y_data[:5000]
X_test = X_data[5000:]
y_test = y_data[5000:]

# unique_data = [list(x) for x in set(tuple(x) for x in XY_data)]
# print len(unique_data)
# this shows that there are only a total of 101 unique X-Y pairs
#
def set_pca_weights():
	'''expects a file with weight on a each new line '''
	pca_results = open('results_PCA.csv')
	pca_weights = np.zeros((56,56))
	i = 0
	for line in pca_results:
		a = float(line.strip())
		pca_weights[i][i] = a
		i +=1
	bias = np.zeros(56)
	return [pca_weights,bias]


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
gate_weights = set_pca_weights()
model.add(Dense(56, input_dim=56, weights=gate_weights))
# model.add(Dense(56, input_dim=56, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(14, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(14, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(14, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=1,
          batch_size=1)
score = model.evaluate(X_test, y_test, batch_size=10)

print "score", score

plot(model, to_file='model.png')

# print "model summary ", model.summary()
# print model.get_config()
# print model.to_json()
weights = model.get_weights()
# for node in  weights[0]:
# 	print node
g = open('keras_weights.txt','w')
g.write(str(weights))
g.close()
