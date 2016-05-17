parameters = {
    'data' : 'color_rus',
    'folder name' : None,
    'subfolder name' : None,
    'target language' : '111', # '111' for color, should correspond to TXT and TXT if 'input sampling responses' == 'corpus'
    #
    'test interval' : 100,
    'n simulations' : 10,
    'length simulation' : 25000,
    #
    'perceptual features' : True,
    'conceptual features' : True,
    'input sampling responses' : 'corpus', # uniform, corpus, siutation
    #
    'pca threshold' : 0.999,
    'leave-one-out' : False,
    # {True,False} TODO: works for GCM and GNB, but for ALCOVE (given dim. weights)??
    'leave target language out' : True,
    'classifier' : 'som', # {gnb, gcm, alcove, som}
    'distance metric' : 'euclidean',
    # {'euclidean', 'manhattan'} only if 'pca preprocessing matrix' = 'distance matrix'
    #
    # GCM PARAMETERS
    'gcm prior' : 'uniform', # or input
    'gcm c' : 1.0,
    'gcm r' : 2.0,
    'gcm p' : 2.0,
    # ALCOVE PARAMETERS
    'alcove c' : 1.0,
    'alcove q' : 1.0,
    'alcove r' : 2.0,
    'alcove phi' : 0.5,
    'alcove lambda_a' : 0.001,
    'alcove lambda_w' : 0.2,
    'alcove attention weight initialization' : 'eigenvalues',
    # {'uniform','eigenvalues'}
    #
    # SOM PARAMETERS
    'som neighborhood function' : 'gaussian',
    'som alpha' : 0.05,
    'som a' : 0.2,
    'som c' : 1.0,
    'som size' : 10,
    'som sigma_0' : 1.0,
    'som lambda_sigma' : 2000,
    'som n pretrain' : 0,
    'som initialization bandwidth' : 0.1,
    'som neighborhood' : 'euclidean', # 'vonneuman'
    #
    # DISCRIMINATION EXPERIMENTS
    'discrimination data' : 'winawer'
    }