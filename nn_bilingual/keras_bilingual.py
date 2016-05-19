import sys, os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.layers import Merge, merge
from keras.layers import Input
import time

np.set_printoptions(threshold=np.nan)

def set_pca_weights(pca_filepath):
	'''expects a file with weight on a each new line '''
	pca_results = open(data_directory+pca_filepath,'r')
	pca_weights = np.zeros((PCA_dimension+1,PCA_dimension+1))
	i = 0
	for line in pca_results:
		a = float(line.strip())
		pca_weights[i][i] = a
		i +=1
	pca_weights[i][i] = 1.0
	# unsure what to set the language gate to start at
	# currently set to 1
	bias = np.concatenate((np.zeros(PCA_dimension),np.ones(1)))
	pca_results.close()
	return [pca_weights,bias]

def load_english_data(english_filepath):
	''' 1. seperate train and test better
		2. change number of iterations for test/train
	'''
	f = open(data_directory+english_filepath,'r')
	XY_data=[]
	X_data = []
	y_data = []
	second_language = [0 for x in range(ger_num_preps)]
	for line in f:
	    data = line.strip().split(',') + [0] + second_language
	    XY_data.append(data)
	    # add one to signify what language
	    # 0 = english
	    # 1 = german
	    X_data.append(data[:PCA_dimension+1])
	    y_data.append(data[PCA_dimension+1:])
	X_train = X_data[:5000]
	y_train = y_data[:5000]
	X_test = X_data[5000:]
	y_test = y_data[5000:]
	return X_train,y_train, X_test, y_test

REAL = True
# REAL = False
english_filepath = 'results_tensor.csv'
pca_filepath = 'results_PCA.csv'
if (REAL == True):
	time_stamp = time.strftime("%m-%d-%Y-%H:%M")
	data_directory = 'data/'
	german_filepath = 'NEED TO FILL IN'
	eng_num_preps = 14
	ger_num_preps = 10
	PCA_dimension = 56
	language_embedding_dimension = 28
else:
	data_directory = 'fake_data/'
	time_stamp = ''
	eng_num_preps = 2
	ger_num_preps = 3
	PCA_dimension = 3
	language_embedding_dimension = 4

X_train, y_train, X_test, y_test = load_english_data(english_filepath)

# unique_data = [list(x) for x in set(tuple(x) for x in XY_data)]
# print len(unique_data)
# this shows that there are only a total of 101 unique X-Y pairs


# _______________first layer___________________#
# language_input = Sequential()
language_input = Input(shape=(PCA_dimension+1,))
# gets the pca over all languages to initialize gates to

gate_weights = set_pca_weights(pca_filepath)

gate_layer = Dense(PCA_dimension+1, activation='linear', input_dim=PCA_dimension+1, weights=gate_weights)(language_input)
# _____________________________________________#

# _____________English Middle Layer____________#
english_representation = Dense(language_embedding_dimension,activation='linear', init='uniform')(gate_layer)

# _____________________________________________#

# _____________English Model_____________#
english_predictions = Dense(eng_num_preps+ger_num_preps, init='uniform', activation='softmax')(english_representation)
english_model = Model(name='english', input=language_input, output=english_predictions)
# _____________________________________________#


# ____________Optimization & Training___________#
sgd = SGD(lr=0.1, decay=1e-6)
english_model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

english_model.fit(X_train, y_train,
          nb_epoch=1,
          batch_size=1)
english_score = english_model.evaluate(X_test, y_test, batch_size=10)
# _____________________________________________#


german_representation = Dense(language_embedding_dimension,activation='linear', init='uniform')(gate_layer)
# bilingual_representation = merge([english_representation,german_representation], mode='concat')

# german_predictions = Dense(eng_num_preps+ger_num_preps, init='uniform', activation='softmax')(bilingual_representation)

# bilingual_predictions = Dense(eng_num_preps+ger_num_preps, weights=english_predictions+german_predictions, activation='softmax')
german_predictions= Dense(eng_num_preps+ger_num_preps, init='uniform', activation='softmax')(german_representation)
merged_predictions = merge([english_predictions,german_predictions], mode='sum')

bilingual_model = Model(name='bilingual', input=language_input, output=merged_predictions)

# print "english_score", english_score

# plot(language_input, to_file='language_input.png')
# plot(english_representation, to_file='english_representation.png')

# print "english_input summary ", english_input.summary()
# print english_input.get_config()
# print english_input.to_json()

def output_experiment(models):
	''' takes a list of models and
	make a new directory with plots of models
	weights of models, and summary of models'''
	directory = time_stamp+'_bilingual_keras_output'
	if not os.path.exists(directory):
	    os.makedirs(directory)

	for model in models:
		weights = model.get_weights()
		filename = model.name + '_weights.txt'
		g = open(directory+'/'+filename,'w')
		g.write(str(weights))
		g.close()
		model.summary()# prints out summaries
		filename = model.name + '_json.txt'
		g = open(directory+'/'+filename,'w')
		g.write(str(model.to_json()))
		g.close()

		plot(model, to_file=directory+'/'+model.name+'.png')
	# weights = bilingual_model.get_weights()
	# g = open('keras_bilingual_weights.txt','w')
	# g.write(str(weights))
	# g.close()

output_experiment([english_model, bilingual_model])
