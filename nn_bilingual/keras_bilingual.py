import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.layers import Merge, merge
from keras.layers import Input
from keras import callbacks
import matplotlib.pyplot as plt
import time
import itertools
import json
from collections import Counter
from itertools import izip
import os, sys
import keras.callbacks
from keras.callbacks import ModelCheckpoint


''' [x]test on 101 unique points ceiling
    [] and test on 71 real ones
    []print out situation mistakes with the sentence / image. number of mistakes made
    []inspect the drop of mistakes
    []run uniform to get other languages sets, and dutch with freq
    [X]numpy heatmaps for the gates
    []numpy heatmaps subtracting 2 things trained.
    []run smaller batches
    []make graphs from paper of top situation answers vs time
    []make inputs all be arrays of integers not list strings
    [] read in number of prepositions per language automatically
    [] introduce callbacks for learning plots
'''
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

def load_language_data(language, data_directory):
  ''' 1. seperate train and test better
      2. change number of iterations for test/train
      dut_x = situation#, 56 floats \n
      dut_y = preposition number \n
  '''
  f = open(data_directory+language+'_x','r')
  g = open(data_directory+language+'_y','r')
  h = open(data_directory+'golden_'+language,'r')
  situation_x_data = {}
  x_train = []
  y_train = []
  x_validation = []
  y_validation =[]
  x_test = []
  y_test = []
  situation_train = []

  for x_line,y_line in izip(f,g):
    x_data = x_line.strip().split(',')
    y_data = y_line.strip().split(',')
    x_data.insert(PCA_dimension+1,0)

    sit_num = int(x_data[0])
    situation_train += [sit_num] # list of all training situations

    situation_x_data[sit_num] = map(float,x_data[1:])
    # print x_data
    x_train += [x_data[1:]]
    y = [0 for x in range(dut_num_preps + ger_num_preps)]
    y[map(int,y_data)[0]] = 1
    y_train += [y]

  #first add language feature
  sit_number_to_preps = make_sit_to_prep(language)
  xy_test_data, test_situation = make_test_data(language,sit_number_to_preps, situation_x_data)
  for xy in xy_test_data:
    x_test += [xy[:-1]]
    y = [0 for x in range(dut_num_preps + ger_num_preps)]
    y[xy[-1:][0]] = 1
    # y[] = 1
    y_test += [y]

  x_validation = x_test
  y_validation = y_test

  #add a bunch of asserts, about sizes, and types of things
  # print len(test_situation[0])

  # counting_situations(situation_train)

  return x_train, y_train, x_validation, y_validation, x_test, y_test, test_situation

def counting_situations(situations):
  g = open(data_directory+language+'_y','r')
  y_s = []
  for y_line in g:
    y_data = y_line.strip().split(',')
    y_s += map(int,y_data)
  preps = []
  for y in y_s:
    preps += [prep_num_to_prep(language,y)]
  c = Counter(zip(situations,preps))
  print sorted(c.items())

def prep_num_to_prep(language, prep_num):
  for key in term_indices[language].keys():
    if(prep_num == term_indices[language][key]):
      return key

def make_test_data(language,situations_to_preps, situation_x_data):
  ''' should this have to read through all lines '''
  xy_test_data = []
  test_situation  = []
  for situation in situations_to_preps.keys():
    xy = situation_x_data[situation]+[situations_to_preps[situation]]
    xy_test_data += [xy]
    test_situation += [situation]

  return xy_test_data, test_situation

def make_sit_to_prep(language):
  test_data = set([]) # 56(PCA)+1(lang)|label
  situations_to_preps = {} # map 71 situations to the correct preposition number
  f = open(data_directory+'golden_'+language,'r')
  for line in f:
    line = line.strip().split(',')
    #line = 63(sit#),over(prep),probability...
    situations_to_preps[int(line[0])] = term_indices[language][line[1]]
    #add asserts about length of data_structures
  return situations_to_preps


def lang_term_number(language, term):
    print term, data[language][term]

def load_term_indices():
    f = open(data_directory+'term_indices')
    data = json.load(f)
    # for language in data:
    #     print language
    #     print data[language]

def output_experiment(models, save_directory, language):
    ''' takes a list of models and
    make a new directory with plots of models
    weights of models, and summary of models'''
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)
    for model in models:
        filepath_weights = save_directory+model.name +'_'+language+'_weights.h5'
        model.save_weights(filepath_weights)

        model.summary()# prints out summaries
        filename = model.name + '_json.txt'
        g = open(save_directory+'/'+filename,'w')
        g.write(str(model.to_json()))
        g.close()
        plot(model, show_shapes=True, to_file=save_directory+'/'+model.name+'.png')

def visualize_gates(model, init_gate, language):
    x_old = init_gate[0].diagonal()
    x_final = model.get_weights()[0].diagonal()
    x_index = [i for i in range(len(x_old)+1)]
    y_index = [0,1,2,3]

    intensity = np.concatenate(([x_old], [x_final], [np.subtract(x_final,x_old)]), axis=0)

    print x_old
    print x_final
    difference = np.subtract(x_final, x_old)
    print difference

    f = open(save_directory+'gates_output','a')
    f.write(language+'\n')
    f.write(str(x_old)+'\n')
    f.write(str(x_final)+'\n')
    f.write(str(difference)+'\n')
    f.write('\n')
    f.close()

    x, y = np.meshgrid(x_index, y_index)

    # plt.pcolormesh(x, y, intensity)
    # plt.colorbar() #need a colorbar to show the intensity scale
    # plt.show() #boom

def model_learn(language, data_directory, save_directory):
  X_train,y_train, X_validation, y_validation, X_test, y_test, test_situation = load_language_data(language, data_directory)

  # _______________first layer___________________#
  # language_input = Sequential()
  language_input = Input(shape=(PCA_dimension+1,))
  # gets the pca over all languages to initialize gates to

  gate_weights = set_pca_weights(pca_filepath)
  init_gate = gate_weights

  gate_layer = Dense(PCA_dimension+1, activation='linear', input_dim=PCA_dimension+1, weights=gate_weights)(language_input)
  # _____________________________________________#

  # _____________English Middle Layer____________#
  english_representation = Dense(language_embedding_dimension,activation='relu', init='uniform')(gate_layer)

  # _____________________________________________#

  # _____________English Model_____________#
  english_predictions = Dense(dut_num_preps+ger_num_preps, init='uniform', activation='softmax')(english_representation)
  english_model = Model(name='english', input=language_input, output=english_predictions)
  # _____________________________________________#


  # ____________Optimization & Training___________#
  sgd = SGD(lr=0.1, decay=1e-6)


  checkpointer = ModelCheckpoint(filepath=save_directory+language+'_weights.{epoch:02d}.hdf5', monitor='val_acc', verbose=1, save_best_only=True)


  english_model.compile(loss='mean_squared_error',
                optimizer=sgd,
                metrics=['accuracy'])


  remote = callbacks.RemoteMonitor(root='http://localhost:9000')

  # english_model.fit(X_train, y_train, nb_epoch=10, batch_size=1)
  english_model.fit(X_train, y_train, batch_size=1, validation_data=(X_validation, y_validation),
      nb_epoch=training_epoch, callbacks=[remote,checkpointer])

  english_score = english_model.evaluate(X_test, y_test, batch_size=1)

  german_representation = Dense(language_embedding_dimension,activation='relu', init='uniform')(gate_layer)
  # bilingual_representation = merge([english_representation,german_representation], mode='concat')

  # german_predictions = Dense(dut_num_preps+ger_num_preps, init='uniform', activation='softmax')(bilingual_representation)

  # bilingual_predictions = Dense(dut_num_preps+ger_num_preps, weights=english_predictions+german_predictions, activation='softmax')
  german_predictions= Dense(dut_num_preps+ger_num_preps, init='uniform', activation='softmax')(german_representation)
  merged_predictions = merge([english_predictions,german_predictions], mode='sum')

  bilingual_model = Model(name='bilingual', input=language_input, output=merged_predictions)

  print "english_score", english_score

  output_experiment([english_model, bilingual_model], save_directory, language)
  visualize_gates(english_model, init_gate, language)

  # plot(language_input, to_file='language_input.png')
  # plot(english_representation, to_file='english_representation.png')

  # print "english_input summary ", english_input.summary()
  # print english_input.get_config()
  # print english_input.to_json()

mode = 'synthetic'
# language = 'dut'
pca_filepath = 'results_PCA.csv'
training_epoch = 200
if (mode == 'real'):
    time_stamp = time.strftime("%m-%d-%Y-%H-%M")
    data_directory = 'gen_data/'
    german_filepath = 'NEED TO FILL IN'
    dut_num_preps = 14
    ger_num_preps = 10
    PCA_dimension = 56
    language_embedding_dimension = 28
elif(mode =='fake'):
    data_directory = 'fake_data/'
    time_stamp = ''
    dut_num_preps = 2
    ger_num_preps = 3
    PCA_dimension = 3
    language_embedding_dimension = 4
else:
    data_directory = 'synthetic_data/'
    time_stamp = ''
    # language = 'lang_1'
    dut_num_preps = 2
    ger_num_preps = 2
    PCA_dimension = 2
    language_embedding_dimension = 4
term_indices = json.load(open(data_directory+'term_indices'))

save_directory = time_stamp+'_bilingual_keras_output/'
if not os.path.exists(save_directory):
      os.makedirs(save_directory)

def main(languages):
  for language in languages:
    model_learn(language, data_directory, save_directory)

if __name__ == "__main__":
  main(sys.argv[1:])


