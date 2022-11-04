#!/usr/bin/env python
# coding: utf-8

# # GMM with MFCC

# Here, we are using **Gaussian Mixture Model (GMM)** as a feature extractor

# In[9]:


import librosa
import numpy as np
from sklearn.mixture import GaussianMixture

y, sr = librosa.load('./p252_002_mic1.flac')
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
delta_mfcc = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(delta_mfcc)

mfcc_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
gmm = GaussianMixture(n_components=16)
gmm.fit(mfcc_features)


# In[13]:


gmm.predict(mfcc_features).shape


# In[ ]:




