import re
import numpy as np
import csv
from collections import defaultdict as dd
#
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#
A = np.array

class data:

    def __init__(self, parameters):
        # set local parameters
        self.dirname = '%s/%s' % (parameters['folder name'], parameters['subfolder name'])
        self.target_language = parameters['target language']
        self.leave_out_language = (None if parameters['leave target language out'] 
                                   else self.target_language)
        self.distance_metric = parameters['distance metric']
        self.pca_threshold = parameters['pca threshold']
        self.conc = parameters['conceptual features']
        self.perc = parameters['perceptual features']
        self.data_folder = parameters['data']
        self.input_sampling_responses = parameters['input sampling responses']
        self.data = parameters['data']
        #
        #
        self.initialize_data()
        dim_w_conc, conceptual_features = self.initialize_conceptual_features()
        dim_w_perc, perceptual_features = self.initialize_perceptual_features()
        self.situations = np.hstack((perceptual_features, conceptual_features))
        # initializes several variables; reads features from files and subsequently combines the
        # perceptual and conceptual features
        self.nF = self.situations.shape[1]
        self.nFP, self.nFC = conceptual_features.shape[1], perceptual_features.shape[1]
        # set number of features (split per perc and conc too)
        self.dim_weights = np.hstack((dim_w_perc, dim_w_conc))
        # set initial dimension weights (for ALCOVE)
        #
        self.P_t = self.initialize_P_t()
        # set corpus probabilities of terms
        self.nT = self.P_t.shape[0]
        # set number of terms for target language
        self.P_s_given_t = normalize(self.CMs[self.target_language].T, norm = 'l1', axis = 1)
        # set conditional probabilities of situations given terms for target languge
        self.max_P_t_given_s = self.CMs[self.target_language].argmax(1)
        self.max_P_t_given_s[self.CMs[self.target_language].sum(1) == 0] = -1
        # set the most likely term given every situation; for unseen situations set to -1
        self.terms = A(sorted(self.term_indices[self.target_language].keys(),
                       key = lambda k : self.term_indices[self.target_language][k]))
        # creates list of terms for target language for quick access.
        self.dump_situations()
        # creates a csv file with the representations used by the model in it

    def initialize_data(self):
        # sets term_indices (per language, languages (set of languages, nS (number of situations)
        # as wel as CMs (count matrices for every language; { language : nS x nT_language }
        with open('%s/elicited_features.csv' % self.data, 'r') as fh:
            self.elicited_features = list(csv.reader(fh))
        self.term_indices = dd(lambda : {})
        self.languages = set()
        self.nS = 0
        CM_constructor = dd(lambda : dd(lambda : dd(float)))
        #
        for language, subject, situation, word in self.elicited_features:
            self.languages.add(language)
            self.nS = np.max([self.nS, int(situation)+1])
            try:
                word_ix = self.term_indices[language][word]
            except KeyError:
                word_ix = self.term_indices[language][word] = len(self.term_indices[language])
            CM_constructor[language][int(situation)][word_ix] += 1.0
        self.CMs = { language : np.zeros((self.nS, len(self.term_indices[language]))) 
                     for language in self.languages}
        for language, v1 in CM_constructor.items():
            for situation, v2 in v1.items():
                for term, count in v2.items():
                    self.CMs[language][situation,term] = count
        return

    def initialize_conceptual_features(self):
        # returns the initial dimension weights of the features (for ALCOVE) as well as the
        # conceptual featureson the basis of crosslinguistic elicitations if
        # conceptual_features == True, or returns an empty array and an empty matrix
        if self.conc == False: return np.zeros((0)), np.zeros((self.nS,0))
        # returns empty matrices if no conceptual features are to be used
        global_DM = np.zeros((self.nS, self.nS))
        for language, cm in self.CMs.items():
            if language == self.leave_out_language: continue
            global_DM += pairwise_distances(normalize(cm, norm = 'l1', axis = 1), 
                                            metric = self.distance_metric)
            global_DM = global_DM / global_DM.max()
        # creates global distance matrix by summing the pairwise distances (according to
        # distance_metric) over all languages (with the exception of the leave_out_language if that
        # variable is set). Then normalizes distance matrix s.t. maximum is 1
        # 
        pca = PCA()
        transformed = pca.fit_transform(global_DM)
        # creates an nS x n Components matrix by applying PCA to the global distance matrix
        #
        cutoff = next(i+1 for i in range(transformed.shape[1])
                      if pca.explained_variance_ratio_[:i].sum() > self.pca_threshold)
        transformed = transformed[:, :cutoff]
        dim_weights = pca.explained_variance_[:cutoff] / pca.explained_variance_.max()
        # obtains a cutoff value; only the first n components are used such that the proportion of
        # the variance captured by the first n components is the lowest n to reach the
        # pca_threshold. The transformed and dim_weights are clipped to this n, and dim_weights is
        # normalized s.t. the maximum value of dim_weights is 1
        #
        range_t = transformed.max(0) - transformed.min(0)
        transformed_centered = (transformed - transformed.mean(0)) / range_t.max() + 0.5
        # centers the transformed values s.t. the mean per feature is 0.5, the feature with the
        # highest eigenvalue has values s.t. max - min = 1 and the other features have proportional
        # ranges according to their eigenvalues. 
        return dim_weights, transformed_centered

    def initialize_perceptual_features(self):
        # returns the initial dimension weights of the features (for ALCOVE) as well as the
        # perceptual features on the basis of crosslinguistic elicitations if
        # perceptual_features == True, or returns an empty matrix
        if self.perc == False:
            return np.zeros(0), np.zeros((self.nS,0))
        else:
            with open('%s/perceptual_features.csv' % self.data_folder, 'r') as fh:
                perceptual = A([A([float(c) for c in row]) for row in csv.reader(fh)])
            range_ = perceptual.max(0) - perceptual.min(0)
            dim_weights = range_ / range_.max()
            perceptual_centered = (perceptual - perceptual.mean(0)) / range_.max() + 0.5
            # centers the perceptual values s.t. the mean per feature is 0.5, the feature with the
            # highest original range now has a range of 1, and the other features have proportional
            # ranges according to their original ranges.
            return dim_weights, perceptual_centered

    def initialize_P_t(self):
        # returns the probabilities of the terms as read off from a frequencies.csv file
        count = np.ones(len(self.term_indices[self.target_language]))
        if self.input_sampling_responses == 'corpus':
            with open('%s/frequencies.csv' % self.data_folder, 'r') as fh:
                freqs = [f for f in csv.reader(fh) if f[0] == self.target_language]
            for language, word, freq in freqs:
                word_ix = self.term_indices[language][word]
                count[word_ix] = float(freq)
        return normalize([count], norm = 'l1')[0]

        
    def dump_situations(self):
        # writes the feature values per situation
        header = ','.join([ 'perc%d' % (i) for i in range(self.nFP) ] +
                          [ 'conc%d' % (i) for i in range(self.nFC) ])
        with open('%s/situations.csv' % self.dirname, 'w') as fh:
            fh.write('%s\n' % header)
            for row in self.situations:
                fh.write('%s\n' % ','.join('%.3e' % np.real(cell) for cell in row))
        return

    def read_discrimination_data(self, parameters):
        # creates an array of 20 stimuli for discrimination experiments.
        self.discrimination_data = parameters['discrimination data']
        if self.discrimination_data == 'winawer':
            start, end = 278, 187
            # takes situations whose Lab values are closest to 20 Winawer stimuli and
            # that have require maxPt|s to be sin/gol; end = 195 if this last constraint is not in
            # place
        elif self.discrimination_data == 'green/blue':
            # start, end = x,y TODO: put in x and y
            pass
        start_v, end_v = self.situations[start], self.situations[end]
        self.discrimination_stimuli = A([start_v - i*(start_v-end_v)/19 for i in range(20)])
        return
