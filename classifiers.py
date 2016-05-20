from sklearn.naive_bayes import GaussianNB
# DELEER from scipy.stats import entropy as kl
# DELEER from scipy.stats import pearsonr
import numpy as np
from collections import Counter
# DELEER import math
# MIGRATE from sklearn.metrics import confusion_matrix as cm
# DELEER import csv
#from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import dill as pickle
from sklearn.metrics import pairwise_distances

A = np.array

class classifier:
    # classifier; contains shared functions between the various categorization models
    
    def __init__(self, data, parameters, simulation):
        # set central variables and parameters
        self.time = 0
        self.data = data
        self.parameters = parameters
        self.simulation = simulation
        self.target_language = parameters['target language']
        self.input_sampling_responses = parameters['input sampling responses']
        self.n_iterations = parameters['length simulation']
        self.len_interval = parameters['test interval']
        #
        self.initialize_model(parameters)
        # model-specific initialization

    def sample_input_item(self):
        # returns a sampled vector of feature-values (reals) for a situation and a term (integer)
        if (self.input_sampling_responses == 'corpus' or 
            self.input_sampling_responses == 'uniform'):
            # sampling on the basis of corpus/uniform term frequencies and P(s|t)
            term = np.random.choice(self.data.nT, 1, p = self.data.P_t)[0]
            situation = np.random.choice(self.data.nS, 1, p = self.data.P_s_given_t[term])[0]
        elif self.input_sampling_responses == 'situation':
            # sampling on the basis of a uniform distribution over situations and P(t|s)
            situation = None
            while (situation == None or self.data.max_P_t_given_s[situation] == -1):
                situation = np.random.choice(self.data.nS)[0]
            term = np.random.choice(self.data.nT, 1, p = self.data.P_t_given_s[situation])[0]
        #
        coordinates = self.data.situations[situation]
        return coordinates, term

    def train(self, test = False, dump = False):
        # training the model for n_iterations iterations, writing away the state of the model
        # every len_interval iterations if dump == True. Runs test() if test == true.
        print('training for simulation %d of %s' % (self.simulation, self.data.dirname))
        d = self.data
        while self.time < self.n_iterations:
            self.time += self.len_interval
            self.fit()
            if dump: self.dump()
            if test: self.test()
        return

    def test(self):
        # tests the whole set of situations and writes both convergence with adult modal naming
        # behavior as well as the distributions over terms per situation to output files.
        # 
        self.development_fh = open('%s/results.csv' % self.data.dirname, 'a')
        self.convergence_fh = open('%s/convergence.csv' % self.data.dirname, 'a')
        #
        posterior = self.predict_terms(self.data.situations)
        predicted_best_term = posterior.argmax(1)
        mpt = self.data.max_P_t_given_s
        predictions = (predicted_best_term == mpt)[mpt != -1]
        self.convergence_fh.write('%d,%d,%.3f\n' % 
                                  (self.simulation, self.time, np.mean(predictions)))
        for i in range(self.data.nS):
            self.development_fh.write('%d,%d,%d,%s\n' % (self.simulation, self.time, i,
                                                    ','.join(['%.3f' % p for p in posterior[i]])))
        self.development_fh.close()
        self.convergence_fh.close()
        return
        #
                                   
class gnb(classifier):
    # Gaussian Naive Bayes
    def initialize_model(self, parameters): 
        self.X, self.Y = [], []
        return

    def fit(self):
        new_X, new_Y = zip(*[self.sample_input_item() for i in range(self.len_interval)])
        self.X.extend(new_X)
        self.Y.extend(new_Y)
        self.classifier = GaussianNB()
        self.classifier.fit(self.X, self.Y)
        return

    def dump(self):
        # pickles the sklearn GaussianNB classifier
        with open('%s/model_%d_%d.p' % (self.data.dirname, self.simulation, self.time),'wb') as fh:
            pickle.dump(self.classifier, fh)
        return

    def load(self, simulation, time):
        # loads the pickled sklearn GaussianNB classifier
        self.simulation = simulation
        self.time = time
        with open('%s/model_%d_%d.p' % (self.data.dirname, simulation, time), 'rb') as fh:
            self.classifier = pickle.load(fh)
        return

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test item i in I from
        # a matrix I x F for I test items each with F features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        posterior_incomplete = self.classifier.predict_proba(test_items)
        posterior = np.zeros((test_items.shape[0], self.data.nT))
        for y_ix, y in enumerate(sorted(set(self.Y))):
            posterior[:,y] += posterior_incomplete[:,y_ix]
        return posterior
        

class gcm(classifier):
    # Generalized Context Model
    # REFERENCE: Nosofsky, R. M. (1987). Attention and learning processes in the identification and
    #   categorization of integral stimuli. Journal of Experimental Psychology: Learning, Memory, 
    #   and Cognition, 13(1), 87-108.
    # TODO: test and check if rightly implemented

    def initialize_model(self, parameters): 
        self.prior_type = parameters['gcm prior']
        self.c = parameters['gcm c']
        self.r = parameters['gcm r']
        self.p = parameters['gcm p']
        self.X, self.Y = [], []#np.zeros((0,self.data.nF)), np.zeros((0))
        #
        self.memorized_summed_eta = np.zeros((self.data.nS, self.data.nT))
        return
    
    def fit(self): 
        new_X, new_Y = zip(*[self.sample_input_item() for i in range(self.len_interval)])
        self.X.extend(new_X)
        self.Y.extend(new_Y)
        return

    def dump(self):
        # pickles the X and Y vectors (situations and terms).
        with open('%s/model_%d_%d.p' % (self.data.dirname, self.simulation, self.time),'wb') as fh:
            pickle.dump((self.X,self.Y), fh)

    def load(self, simulation, time):
        self.simulation = simulation
        self.time = time
        with open('%s/model_%d_%d.p' % (self.data.dirname, simulation, time), 'rb') as fh:
            self.X, self.Y = pickle.load(fh) 
        return

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test item i in I from
        # a matrix I x F for I test items each with F features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        #
        if test_items.shape[0] == self.data.nS:
            X,Y = A(self.X[-self.len_interval:]), A(self.Y[-self.len_interval:])
        else: X,Y = A(self.X), A(self.Y)
        #
        if self.prior_type == 'input':
            counts = np.zeros((test_items.shape[0], self.data.nT))
            counts_incomplete = Counter(Y)
            for y,v in counts_incomplete.items(): counts[:,y] = v
            b = normalize(counts, norm = 'l1', axis = 1)
            #counts_complete = A([counts_incomplete[y] for y in range(self.data.nT)]).astype('float')
            #b = normalize([counts_complete], norm = 'l1')[0]
            #b = np.histogram(self.Y, bins = range(self.data.nT+1), density = True)[0]
        elif self.prior_type == 'uniform':
            b = normalize(np.ones((test_items.shape[0], self.data.nT)), norm = 'l1', axis = 1)
            b = np.ones((test_items.shape[0], self.data.nT))
        #
        #self.a_dim = np.ones(self.X[0].shape[0])
        self.a_dim = np.ones((X.shape[0], self.data.nF))
        #
        # d_abs = np.abs( self.X  - test_item )
        # TODO this is suboptimal -- X and Y shd be np arrays from getgo
        # distances = self.c * np.power(np.sum(np.power(self.a_dim * d_abs, self.r), 1), 1.0/self.r)
        distances = A([self.c*np.linalg.norm(self.a_dim * (X-test_items[i]), ord=self.r,axis=1)
                       for i in range(test_items.shape[0])])
                       # TODO X is subopt, shd be np.A
        etas = np.exp(-np.power(distances, self.p))
        #
        if test_items.shape[0] == self.data.nS:
            summed_eta = self.memorized_summed_eta.copy()
        else: summed_eta = np.zeros((test_items.shape[0], self.data.nT))
        for i in range(test_items.shape[0]):
            for y,a in zip(Y.astype('int'),etas[i]): summed_eta[i][y] += a
        if test_items.shape[0] == self.data.nS:
            self.memorized_summed_eta = summed_eta.copy()
        #
        return normalize(b * summed_eta, norm = 'l1', axis = 1)
        #return (b * summed_eta) / np.sum(b * summed_eta)
 
class alcove(classifier):
    # ALCOVE
    # REFERENCE: Kruschke, J. K. (1992). ALCOVE: An exemplar-based connectionist model of category 
    #   learning. Psychological Review, 99(1), 22-44. 

    def initialize_model(self, parameters):
        # parameters
        self.a_dim_initialization = parameters['alcove attention weight initialization']
        self.lambda_w = parameters['alcove lambda_w']
        self.lambda_a = parameters['alcove lambda_a']
        self.c = parameters['alcove c']
        self.q = parameters['alcove q']
        self.r = parameters['alcove r']
        self.phi = parameters['alcove phi']
        # initializes dimension weights a_dim, either as uniform or as set to the relative
        # importance of the features in the data.
        if self.a_dim_initialization == 'uniform':
            self.a_dim = np.ones(self.data.situations.shape[1])/self.data.situations.shape[1]
        elif self.a_dim_initialization == 'eigenvalues':
            self.a_dim = self.data.dim_weights.copy()
        self.hidden = A([])
        self.w = A([])
        self.out = A([])
        # here I deviate from Kruschke's formulation -- my implementation starts with incrementally
        # growing network, because otherwise weights self.w between hidden layer self.hidden and 
        # output layer self.out will be negatively affected without the elements in self.hidden and
        # self.out being observed. The network `grows' incrementally with the self.update_nodes()
        # function. TODO: implement original formulation and parametrize.   
    
    def fit(self):
        # trains the model for len_interval input items
        for i in range(self.len_interval):
            a_in, k = self.sample_input_item()
            self.update_nodes(a_in, k)
            a_hid, a_out = self.get_activations(a_in)
            self.backprop(a_in, a_hid, a_out, k)
        return

    def update_nodes(self, a_in, k):
        # makes the network's hidden and output layer grow. See comment in initialize_model()
        w_init = 0.0
        new_k = k not in self.out
        new_a_in = not any(np.all(a_in == x) for x in self.hidden)
        no_a_in = self.hidden.shape[0] == 0
        #
        if no_a_in:
            self.hidden = A([a_in])
            self.out = np.concatenate((self.out,A([k])), axis = 0)
            self.w = A([A([w_init])])
        else:
            if new_k:
                new_row = A([A([np.random.uniform(-0,0)]) for k_ in range(self.w.shape[1])]).T
                self.out = np.concatenate((self.out,A([k])), axis = 0)
                self.w = np.concatenate((self.w, new_row), axis = 0)
            if new_a_in:
                new_col = A([A([np.random.uniform(-0,0) for h_ in range(self.out.shape[0])])]).T
                self.hidden = np.concatenate((self.hidden, [a_in]), axis = 0)
                self.w = np.concatenate((self.w, new_col), axis = 1)
        return
        
    def get_activations(self, a_in):
        # calculates the activation values for the hidden layer and the output layer of ALCOVE
        # given an input signal a_in
        a_hid = np.exp(-self.c * np.power(np.sum(np.power(self.a_dim * (self.hidden-a_in), self.r),
                                                 axis = 1), self.q/self.r))
        # a_hid = e^(-c(SUM_i(a_dim_i * |h_ji - a_in_i|)^r)^(q/r)
        a_out = np.sum(a_hid * self.w, axis = 1)
        # a_out = SUM_j(w_kj * a_hid_j)
        return a_hid, a_out

    def backprop(self, a_in, a_hid, a_out, k):
        # backpropagates the error signal on the basis of the activation values over the hidden
        # layer and the output layer, given a correct category k and an input signal a_in
        t_k = A([(max(1, a_out[ix]) if k_ == k else min(-1, a_out[ix]))
                 for ix,k_ in enumerate(self.out)])
        # teacher values
        error = t_k - a_out
        delta_w = self.lambda_w * np.outer(error, a_hid)
        delta_a_dim = (-self.lambda_a) * np.sum((np.sum(self.w.T * error, axis = 1) * a_hid)
                                                * (self.c * np.abs(self.hidden-a_in)).T, 1)
        # calculate deltas
        self.w += np.nan_to_num(delta_w)
        # update w
        self.a_dim += np.nan_to_num(delta_a_dim)
        self.a_dim[self.a_dim < 0.0] = 0.0
        # update a_dim and clip negative values

    def dump(self):
        # pickles the layers of the neural network
        with open('%s/model_%d_%d.p' % (self.data.dirname, self.simulation, self.time),'wb') as fh:
            pickle.dump((self.a_dim, self.hidden, self.out, self.w), fh)
        return

    def load(self, simulation, time):
        # loads the pickled NN layers
        self.simulation = simulation
        self.time = time
        with open('%s/model_%d_%d.p' % (self.data.dirname, simulation, time), 'rb') as fh:
            self.a_dim, self.hidden, self.out, self.w = pickle.load(fh) 
        return

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test item i in I from
        # a matrix I x F for I test items each with F features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        e_a_outs = []
        for a_in in test_items:
            a_hid, a_out = self.get_activations(a_in)
            e_a_out = np.nan_to_num(np.exp(self.phi * a_out))
            e_a_outs.append(e_a_out)
        normalized = normalize(A(e_a_outs), norm = 'l1', axis = 1)
        normalized_all_terms = np.zeros((test_items.shape[0], self.data.nT))
        for i, k in enumerate(self.out):
            normalized_all_terms[:,k] = normalized[:,i]
        return normalized_all_terms
        #return normalize(e_a_outs, norm = 'l1', axis = 1).T[self.out.argsort()].T
    
class som(classifier):
    # Self-Organizing Map
    # REFERENCE: H Ritter, T Kohonen (1989). Self-organizing semantic maps. Biological cybernetics 
    #    61(4), 241-254
    # REFERENCE: T. Kohonen (2001): Self-Organizing Maps. Third Edition. Springer

    def initialize_model(self, parameters):
        # parameters
        self.init_bandwidth = parameters['som initialization bandwidth']
        self.size = parameters['som size']
        self.alpha = parameters['som alpha']
        self.a = parameters['som a']
        self.c = parameters['som c']
        self.lambda_sigma = parameters['som lambda_sigma']
        self.sigma_0 = parameters['som sigma_0']
        self.n_pretrain = parameters['som n pretrain']
        self.neighborhood = parameters['som neighborhood']
        #
        # initialize MAP
        self.size_y = self.size_x = self.size
        term_map = np.zeros((self.size_x, self.size_y, self.data.nT))#[:,:,:]
        property_map = (((np.random.rand(self.size_x, self.size_y, self.data.nF) - 0.5 ) *
                         self.init_bandwidth) + 0.5) #[:,:,:]
        self.map = np.concatenate((term_map, property_map), axis = 2)
        self.indices = A([A([A([i,j]) for j in range(self.size_y)]) for i in range(self.size_x)])
        # FUTURE: rectangular and growing maps
        #
        # pretrain on just property features
        self.pretrain()
        #
        return

    def pretrain(self):
        # train the SOM on just property features
        for i in range(self.n_pretrain):
            x, _y = self.sample_input_item()
            self.time += 1
            input_item = self.get_input_item(features = x, term = None)
            self.update_map(input_item, self.time)
        self.time = 0
        return

    def fit(self):
        # train the SOM on input items
        for i in range(self.len_interval):
            x, y = self.sample_input_item()
            input_item = self.get_input_item(features = x, term = y)
            self.update_map(input_item, (self.n_pretrain + self.time - self.len_interval + i ))

    def dump(self):
        # pickles the SOM
        with open('%s/model_%d_%d.p' % (self.data.dirname, self.simulation, self.time),'wb') as fh:
            pickle.dump(self.map.astype('float16'), fh)

    def load(self, simulation, time):
        # loads a pickled SOM
        self.simulation = simulation
        self.time = time
        with open('%s/model_%d_%d.p' % (self.data.dirname, simulation, time), 'rb') as fh:
            self.map = pickle.load(fh)


    def get_input_item(self, features, term = None):
        # on the basis of a string of features and a term, returns one vector, combining a one-hot
        # distribution (with the hot bit set to a) and the feature string
        term_str = self.data.terms[term] if term != None else ''
        return np.hstack(( self.a * (self.data.terms == term_str), features))

    def update_map(self, input_item, time):
        # updates the map with an input item
        sigma = self.sigma_0 * np.exp(-(time / self.lambda_sigma))
        bmu_ix = self.get_bmu_ix(input_item)
        h = np.exp(-self.get_grid_distance(bmu_ix) / (2 * np.power(sigma, 2) ))
        # the formulation with (2*sigma^2) a.o.t. sigma^2 comes from Kohonen (2001), p. 111
        h = h[..., None] * np.ones((self.data.nT + self.data.nF))
        self.map = self.map + self.alpha * h * (-self.map + input_item)

    def get_bmu_ix(self, input_item, ignore_terms = False):
        # gets the map index of the best matching unit
        f = self.data.nT * ignore_terms
        D = np.linalg.norm(self.map[:,:,f:] - input_item[f:], ord = 2, axis = 2)
        return np.unravel_index(D.argmin(), self.map.shape[:2])

    def get_grid_distance(self, bmu_ix, neighborhood = 'euclidean'):
        # gets values for each cell of the map, given a center of activation at bmu_ix
        if neighborhood == 'euclidean': return np.sum(np.power(self.indices - bmu_ix, 2), 2) 
        elif neighborhood == 'vonneuman':
            h = np.zeros(self.indices.shape[:2])
            h[bmu_ix] = 1
            h[bmu_ix[0]-1, bmu_ix[1]] = h[bmu_ix[0]+1, bmu_ix[1]] = 1
            h[bmu_ix[0], bmu_ix[1]-1] = h[bmu_ix[0], bmu_ix[1]+1] = 1 
            return h          

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test item i in I from
        # a matrix I x F for I test items each with F features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        bmu_ixx = []
        for test_item in test_items:
            input_item = self.get_input_item(features = test_item, term = None)
            bmu_ix = self.get_bmu_ix(input_item, True)
            bmu_ixx.append(bmu_ix)
        term_distributions = A([self.map[i,j,:self.data.nT] for i,j in bmu_ixx])
        return normalize(term_distributions, norm = 'l1', axis = 1)

    def get_term_map(self):
        # returns a matrix the size of the SOM with the most likely term in every cell
        return A([A([('%s   ' % (self.data.terms[np.argmax(self.map[i][j][:self.nterm])]
                                 if np.sum(self.map[i,j,:self.nterm]) > 0 else '[]'))[:3]
                     for j in range(self.map.shape[1])]) for i in range(self.map.shape[0])])

    def discriminate(self):
        # calculates the between-cell map distance for the BMUs of an array of stimuli
        # used in Beekhuizen & Stevenson 2016
        testset = A([ self.get_input_item(t) for t in self.data.discrimination_stimuli ])
        positions = A([ A(self.get_bmu_ix(t)) for t in testset ])
        distances = pairwise_distances(positions, metric = 'euclidean')
        dn, tn = self.data.dirname, self.data.discrimination_data
        terms = self.data.terms[self.predict_terms(self.data.discrimination_stimuli).argmax(1)]
        with open('%s/discrimination_terms_%s.csv' % (dn, tn), 'a') as o:
            for i,t in enumerate(terms):
                o.write('%d,%d,%d,%s\n' % (self.simulation, self.time, i, t))
        with open('%s/confusability_%s.csv' % (dn, tn), 'a') as o:
            for i in range(distances.shape[0]):
                for j in range(i+1, distances.shape[0]):
                    o.write('%d,%d,%d,%d,%s,%s,%.3f\n' %
                            (self.simulation, self.time, i, j, terms[i], terms[j], distances[i,j]))
        return
