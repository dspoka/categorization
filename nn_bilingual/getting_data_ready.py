import sys
from itertools import izip
h = open('results_tensor2.csv','w')

single_label = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0']
with open("results_X.csv") as features, open("results_Y.csv") as labels:
    for x, y in izip(features, labels):
        x =  x.split()
        y = int(y)
        single_label[y] = '1'
        h.write(','.join(x+single_label+['\n']))
        single_label[y] = '0'
