# a cup on table
# b bandaid on leg
# c picture on wall
# d apple on branch
# e ribbon on candle
# f apple in boydwalkeri

# English
# a on
# b on
# c on
# d on
# e on
# f in

# Dutch
# a op
# b op
# c aan
# d aan
# e om
# f in
#
# Spanish
# a en
# b en
# c en
# d en
# e en
# f en
#


# synthetic data,
# couple papers,
# data from the hermann,
# error analysisff
# per time. n preposition * n preposition confusion matrix counts

import numpy as np
import os
import json

# 0,0          1,0
# |     |     |
# |  1  |  2  |
# _____________
# |     |     |
# |  3  |  4  |
# 0,1          1,1

# syn_lang_1:
# 1: x
# 2: y
# 3: x
# 4: y

# syn_lang_1:
#   1 2 3 4 5 (synthetic Language)
# 1|a|c|e|g|i
# 2|b|c|f|h|j
# 3|a|d|f|h|k
# 4|b|d|e|h|j

synthetic_directory = 'synthetic_data/'
if not os.path.exists(synthetic_directory):
      os.makedirs(synthetic_directory)

synthetic_term_indices = {'lang_1' : {'a' : 0 ,'b' : 1}, 'lang_2' : {'c' : 0,'d' : 1}, 'lang_3' : {'e' : 0,'f' : 1}, 'lang_4' : {'g' : 0,'h' : 1}, 'lang_5' : {'i' : 0,'j' : 1,'k' : 2}}
json.dump(synthetic_term_indices, open(synthetic_directory+'term_indices','w'))

def make_synthetic_input():
    np.random.seed(seed=0)
    temps = []
    situations = []
    langs = ['1','2','3','4','5']
    lang_1 = []
    lang_2 = []
    lang_3 = []
    lang_4 = []
    lang_5 = []
    synthetic_langs = [lang_1,lang_2,lang_3,lang_4,lang_5]
    for i in range(100):
        temp = np.random.rand(1,2)
        if(temp[0][0] < .5 and temp[0][1] < .5):
            # 1 quadrant
            lang_1 += ['a']
            lang_2 += ['c']
            lang_3 += ['e']
            lang_4 += ['g']
            lang_5 += ['i']
        elif(temp[0][0] < .5 and temp[0][1] > .5):
            # 3 quadrant
            lang_1 += ['a']
            lang_2 += ['d']
            lang_3 += ['f']
            lang_4 += ['h']
            lang_5 += ['k']
        elif(temp[0][0] > .5 and temp[0][1] > .5):
            # 4 quadrant
            lang_1 += ['b']
            lang_2 += ['d']
            lang_3 += ['e']
            lang_4 += ['h']
            lang_5 += ['j']
        else:
            # quadrant 2
            lang_1 += ['b']
            lang_2 += ['c']
            lang_3 += ['f']
            lang_4 += ['h']
            lang_5 += ['j']
        # print i, temp[0][0], temp[0][1], lang_1
        situations += [str(i+1)]
        temps += [temp[0]]
        # print i, temp[0][0], temp[0][1]
        # print lang_1
        # f = open(synthetic_directory+lang_1+'_x','w')
    for lang in langs:
        f = open(synthetic_directory+'lang_'+lang+'_x', 'w')
        for a,b in zip(situations,temps):
            f.write(a+','+str(b[0])+','+str(b[1])+'\n')
        f.close()

    i = 1
    for lang in synthetic_langs:
        f = open(synthetic_directory+'golden_lang_'+str(i), 'w')
        g = open(synthetic_directory+'lang_'+str(i)+'_y', 'w')
        for a,b in zip(situations, lang):
            f.write(a+','+b+'\n')
            g.write(str(synthetic_term_indices['lang_'+str(i)][b])+'\n')
        i += 1
        f.close()
        g.close()

    for lang in langs:
        f = open(synthetic_directory+'lang_'+lang+'_situation', 'w')
        f.write("\n".join(situations))
        f.close()

        # f = open(synthetic_directory+'golden_lang'+lang, 'w')
        # f.write("\n".join(situations))
        # f.close()
make_synthetic_input()