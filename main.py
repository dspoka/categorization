from classifiers import gcm, gnb, alcove, som
from data import data
import parameters
import numpy as np
import sys
import multiprocessing
import os
#
import cPickle as pickle

def experiment_VI(params, folder_name, pool_size=3):
    params['classifier'] = 'som'
    params['folder name'] = '/Users/barendbeekhuizen/Desktop/%s' % folder_name
    param_combos = [[sam, perc, conc, d]
                    for sam in ['corpus', 'uniform'] for perc in [True,False] for conc in [True,False]
                    for d in ['color_rus', 'color_eng'] if not (perc == False and conc == False)]
    features = ['input sampling responses', 'perceptual features', 'conceptual features', 'data']
    shorthands = ['sam', 'perc', 'conc', 'data']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos]
    #pool = multiprocessing.Pool(processes = pool_size)
    #As = pool.map(train, arguments)
    #As = pool.map(test, arguments)
    #As = pool.map(discrimination_experiment, arguments)
    #pool.close()
    for i in arguments:
        train(i)
        test(i)
        discrimination_experiment(i)
    
    return

def experiment_I_III(params, folder_name, pool_size=3):
    params['classifier'] = 'som'
    params['folder name'] = '/Users/barendbeekhuizen/Desktop/%s' % folder_name
    param_combos = [[sam, perc, conc, d, alpha, a, size, lamSig, pretrain]
                    for sam in ['corpus', 'uniform'] for perc in [True,False] for conc in [True,False]
                    for d in ['color_rus', 'color_eng'] 
                    for alpha in [0.05, 0.2] for a in [0.05, 0.2] for size in [6,8,10] 
                    for lamSig in [1000,5000] for pretrain in [0,1000] if not (perc == False and conc == False)]
    features = ['input sampling responses', 'perceptual features', 'conceptual features', 'data',
                'som alpha', 'som a', 'som size','som lambda_sigma','som n pretrain']
    shorthands = ['sam','perc','conc','data','alpha','a','size','lambdasigma','pretrain']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos]
    pool = multiprocessing.Pool(processes = pool_size)
    As = pool.map(train, arguments)
    As = pool.map(test, arguments)
    As = pool.map(discrimination_experiment, arguments)
    pool.close()
    return

def experiment_VIII(params, folder_name, pool_size=3):
    params['classifier'] = 'som'
    params['folder name'] = '%s' % folder_name
    params['data'] = 'color_eng'
    param_combos = [[sam, perc, conc]
                    for sam in ['corpus', 'uniform'] for perc in [True,False] for conc in [True,False]
                    if not (perc == False and conc == False)]
    features = ['input sampling responses', 'perceptual features', 'conceptual features']                
    shorthands = ['sam','perc','conc']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos]
    pool = multiprocessing.Pool(processes = pool_size)
    As = pool.map(train, arguments)
    As = pool.map(test, arguments)
    As = pool.map(discrimination_experiment, arguments)
    pool.close()
    return

def experiment_IV(params, folder_name, pool_size=3):
    params['classifier'] = 'som'
    params['folder name'] = '%s' % folder_name
    param_combos = [[sam, perc, conc, d, cutoff, dm]
                    for sam in ['corpus', 'uniform'] for perc in [True,False] for conc in [True,False]
                    for d in ['color_rus', 'color_eng'] for cutoff in [0.9, 0.99, 0.999]
                    for dm in ['euclidean', 'manhattan'] if not (perc == False and conc == False)]
    features = ['input sampling responses', 'perceptual features', 'conceptual features', 'data',
                'pca threshold', 'distance metric']
    shorthands = ['sam','perc','conc','data','cutoff', 'dm']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos]
    pool = multiprocessing.Pool(processes = pool_size)
    As = pool.map(train, arguments)
    As = pool.map(test, arguments)
    As = pool.map(discrimination_experiment, arguments)
    pool.close()
    return

def experiment_V_VII(params, folder_name, pool_size=3):
    params['folder name'] = '%s' % folder_name
    param_combos = [[sam, perc, conc, d, classifier, init]
                    for sam in ['corpus', 'uniform'] for perc in [True,False] for conc in [True,False]
                    for d in ['color_rus', 'color_eng'] for classifier in ['alcove','gcm', 'gnb'] 
                    for init in ['eigenvalues', 'uniform'] 
                    if (not (perc == False and conc == False)) and (init == 'uniform' or classifier == 'alcove')]
    features = ['input sampling responses', 'perceptual features', 'conceptual features', 'data',
                'classifier', 'alcove attention weight initialization']
    shorthands = ['sam','perc','conc','data','classifier','init']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos]
    pool = multiprocessing.Pool(processes = pool_size)
    As = pool.map(train, arguments)
    As = pool.map(test, arguments)
    pool.close()
    return
    
def experiment_design_som(params):
    params['folder name'] = '/Users/barendbeekhuizen/Desktop/%s' % sys.argv[1]  
    #"""
    param_combos = [[sam, perc, conc, d, pretrain]
                    for sam in ['corpus', 'uniform']
                    for perc in [True,False]
                    for conc in [True,False]
                    for d in ['color_rus']
                    for pretrain in [10000, 2000, 0]
                    if not (perc == False and conc == False)]
    features = ['input sampling responses', 'perceptual features', 'conceptual features', 
                'data', 'som n pretrain']
    shorthands = ['sam', 'perc', 'conc', 'data', 'pretrain']
    #"""
    #param_combos = [[perc, conc] for perc in [True,False] for conc in [True,False] if perc or conc]
    #features = ['perceptual features', 'conceptual features']
    #shorthands = ['perc', 'conc']
    arguments = [(param_combo, params, features, shorthands) for param_combo in param_combos[:2]]
    #
    # non-multithreading processing -- better for debugging
    #for i in arguments: train(i)
    #for i in arguments: test(i)
    #for i in arguments: discrimination_experiment(i)
    # multithreading processing -- better for speed
    pool = multiprocessing.Pool(processes = int(sys.argv[2]))
    #As = pool.map(train_and_test, arguments)
    #As = pool.map(train, arguments)
    #As = pool.map(test, arguments)
    As = pool.map(discrimination_experiment, arguments)


def train(arguments):
    param_combo, params, features, shorthands = arguments
    params = { k:v for k,v in params.items() }
    params['subfolder name'] = '_'.join(['%s_%s'%(sh,pv) for sh,pv in zip(shorthands,param_combo)])
    for pv, fe in zip(param_combo, features): params[fe] = pv
    classifiers = {'gcm':gcm, 'gnb':gnb, 'alcove':alcove, 'som':som}
    classifier = classifiers[params['classifier']]
    dirname = '%s/%s' % (params['folder name'], params['subfolder name'])
    os.makedirs(dirname)
    pickle.dump(params, open('%s/parameters.p' % dirname, 'wb'))
    #
    d = data(params)
    for simulation in range(params['n simulations']):
        c = classifier(d, params, simulation)
        c.train(dump = True, test = False)

def test(arguments):
    param_combo, params, features, shorthands = arguments
    params = { k:v for k,v in params.items() }
    params['subfolder name'] = '_'.join(['%s_%s'%(sh,pv) for sh,pv in zip(shorthands,param_combo)])
    for pv, fe in zip(param_combo, features): params[fe] = pv
    classifiers = {'gcm':gcm, 'gnb':gnb, 'alcove':alcove, 'som':som}
    classifier = classifiers[params['classifier']]
    d = data(params)
    for simulation in range(params['n simulations']):
        c = classifier(d, params, simulation)
        time = 0
        while time < params['length simulation']:
            time += params['test interval']
            c.load(simulation, time)
            c.test()

def discrimination_experiment(arguments):
    param_combo, params, features, shorthands = arguments
    params = { k:v for k,v in params.items() }
    params['subfolder name'] = '_'.join(['%s_%s'%(sh,pv) for sh,pv in zip(shorthands,param_combo)])
    for pv, fe in zip(param_combo, features): params[fe] = pv
    classifiers = {'gcm':gcm, 'gnb':gnb, 'alcove':alcove, 'som':som}
    classifier = classifiers[params['classifier']]
    d = data(params)
    d.read_discrimination_data(params)
    for simulation in range(params['n simulations']):
        c = classifier(d, params, simulation)
        time = 0
        while time < params['length simulation']:
            time += params['test interval']
            c.load(simulation, time)
            c.discriminate()
    

def train_and_test(arguments):
    param_combo, params, features, shorthands = arguments
    params = { k:v for k,v in params.items() }
    params['subfolder name'] = '_'.join(['%s_%s'%(sh,pv) for sh,pv in zip(shorthands,param_combo)])
    for pv, fe in zip(param_combo, features): params[fe] = pv
    classifiers = {'gcm':gcm, 'gnb':gnb, 'alcove':alcove, 'som':som}
    classifier = classifiers[params['classifier']]
    dirname = '%s/%s' % (params['folder name'], params['subfolder name'])
    os.makedirs(dirname)
    pickle.dump(params, open('%s/parameters.p' % dirname, 'wb'))
    #
    d = data(params)
    for simulation in range(params['n simulations']):
        c = classifier(d, params, simulation)
        c.train(dump = False, test = True)


def main():
    params = parameters.parameters
    experiment_V_VII(params, sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    main()
