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

def load_language_data(language):
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

  for x_line,y_line in izip(f,g):
    x_data = x_line.strip().split(',')
    y_data = y_line.strip().split(',')
    x_data.insert(PCA_dimension+1,0)

    sit_num = int(x_data[0])
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

  print len(x_train)
  print len(y_train)
  print len(x_validation)
  print len(y_validation)
  print len(x_test)
  print len(y_test)
  print len(test_situation)
  print
  print len(x_train[0])
  print len(y_train[0])
  print len(x_validation[0])
  print len(y_validation[0])
  print len(x_test[0])
  print len(y_test[0])
  # print len(test_situation[0])

  return x_train, y_train, x_validation, y_validation, x_test, y_test, test_situation

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

REAL = True
# REAL = False
language = 'dut'
pca_filepath = 'results_PCA.csv'
training_epoch = 1
if (REAL == True):
    time_stamp = time.strftime("%m-%d-%Y-%H:%M")
    data_directory = 'gen_data/'
    german_filepath = 'NEED TO FILL IN'
    dut_num_preps = 14
    ger_num_preps = 10
    PCA_dimension = 56
    language_embedding_dimension = 28
else:
    data_directory = 'fake_data/'
    time_stamp = ''
    dut_num_preps = 2
    ger_num_preps = 3
    PCA_dimension = 3
    language_embedding_dimension = 4

term_indices = json.load(open(data_directory+'term_indices'))
X_train,y_train, X_validation, y_validation, X_test, y_test, test_situation = load_language_data(language)


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
english_model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


remote = callbacks.RemoteMonitor(root='http://localhost:9000')

# english_model.fit(X_train, y_train, nb_epoch=10, batch_size=1)
english_model.fit(X_train, y_train, batch_size=1, validation_data=(X_validation, y_validation),
    nb_epoch=training_epoch, callbacks=[remote])

english_score = english_model.evaluate(X_test, y_test, batch_size=1)

german_representation = Dense(language_embedding_dimension,activation='relu', init='uniform')(gate_layer)
# bilingual_representation = merge([english_representation,german_representation], mode='concat')

# german_predictions = Dense(dut_num_preps+ger_num_preps, init='uniform', activation='softmax')(bilingual_representation)

# bilingual_predictions = Dense(dut_num_preps+ger_num_preps, weights=english_predictions+german_predictions, activation='softmax')
german_predictions= Dense(dut_num_preps+ger_num_preps, init='uniform', activation='softmax')(german_representation)
merged_predictions = merge([english_predictions,german_predictions], mode='sum')

bilingual_model = Model(name='bilingual', input=language_input, output=merged_predictions)

print "english_score", english_score

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
        plot(model, show_shapes=True, to_file=directory+'/'+model.name+'.png')

def visualize_gates():
    x_old = init_gate[0].diagonal()
    x_final = english_model.get_weights()[0].diagonal()
    x_index = [i for i in range(len(x_old)+1)]
    y_index = [0,1,2,3]

    intensity = np.concatenate(([x_old], [x_final], [np.subtract(x_final,x_old)]), axis=0)

    print x_old
    print x_final
    print np.subtract(x_final, x_old)

    x, y = np.meshgrid(x_index, y_index)

    plt.pcolormesh(x, y, intensity)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom


# output_experiment([english_model, bilingual_model])
# visualize_gates()


right_answer = [0 for x in range(71)]
second_answer = [0 for x in range(71)]
wrong_answer = [0 for x in range(71)]

y_test = np.array(y_test)
for x,y,z in zip(english_model.predict(X_test, batch_size=1), y_test, test_situation):
  # print x, y
  prediction = np.where(x==x.max())[0][0]
  prediction_2 = np.argsort(x)[::-1][1]
  correct_answer = np.where(y==y.max())[0][0]
  if(prediction == correct_answer):
    right_answer[z-1] += 1
  # elif(prediction_2 == correct_answer):
  #   print " " + str(prediction), correct_answer
  #   second_answer[z-1] += 1
  else:
    # print " " + str(prediction), correct_answer
    wrong_answer[z-1] += 1
  for key, value in term_indices[language].iteritems():
    if value == prediction:
      break
  print z,key #this prints out situation, preposition predicted

N = 71

ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, right_answer, width, color='g')
rects2 = ax.bar(ind + width, wrong_answer, width, color='r')
rects3 = ax.bar(ind + width + width, second_answer, width, color='y')

# add some text for labels, title and axes ticks
ax.set_xticks(ind + width)
ax.set_xticklabels([i+1 for i in range(71)])
ax.legend((rects1[0], rects2[0],rects3[0]), ('Right', 'Wrong', '2nd right'))

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.show()

print "sums of right, 2nd, wrong"
print sum(right_answer)
print sum(second_answer)
print sum(wrong_answer)

print
for i in range(len(second_answer)):
  if(second_answer[i]+wrong_answer[i] != 0):
    print i

# def main():

# if __name__ == "__main__":
#     main()
