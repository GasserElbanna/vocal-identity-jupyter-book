#!/usr/bin/env python
# coding: utf-8

# # openSMILE Fetaures

# [openSMILE](https://audeering.github.io/opensmile/) is short for **open**-**S**ource **M**edia **I**nterpretation by **L**arge feature-space **E**xtraction. It is a toolkit to extract DSP-based features.

# OpenSMILE generates low-level descriptors (LLDs) and compute functionals on LLD contours. The most commonly used feature sets are **eGeMAPS** and **ComParE** both comprises 88 and 6373 features, respectively.

# We used these feature sets to extract handcrafted features from audios resampled to 16 kHz.

# In[4]:


import librosa
import opensmile

egemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

y, sr = librosa.load('./p252_002_mic1.flac', sr=16000)
egemaps_embeddings = egemaps.process_signal(y, sr).to_numpy().flatten()
compare_embeddings = compare.process_signal(y, sr).to_numpy().flatten()

print(f'The size of eGeMAPS embeddings is {egemaps_embeddings.shape[0]}')
print(f'The size of ComParE embeddings is {compare_embeddings.shape[0]}')

