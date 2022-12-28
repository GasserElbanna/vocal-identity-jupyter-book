#!/usr/bin/env python
# coding: utf-8

# # Pitch-shifting Speech Experiment

# In this section, we explore the impact of pitch-shifting on voice identity processing.

# ## Literature Review:

# [Looney and Gaubitch (2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9414890):
# 
# 
# The authors compared voice-modification detection performance between humans and machines. Wherefore, they shifted the pitch of an english language speech corpus ranging from -600 to 500 cents using the *sox* implementation of WSOLA algorithm.
# They used a feature set that was capable of yielding comparable results with humans for anomaly detection. The aforementioned feature set comprised source-filter model parameters (e.g. F0, formants and LPC residual structure). Finally, they proposed an unsupervised speech anomaly detection (SAnD) scheme.

# ### Research Questions:
# 1) How do different models perceive changes in pitch?
# 2) Which models would be invariant to pitch-shifting?
# 3) How could pitch-shifting impact the models ability to predict identity?

# ## Dataset Description:

# The dataset used in this experiment is [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443). It comprises speech utterances from 110 english speakers with different accents. However, we created a mini version of this dataset by selecting only 10 speakers (5 Males / 5 Females). Each speaker reads about 400 scentences.

# The dataset was processed by undersampling to 16 kHz to be compatible with the pre-trained self-supervised models. The pitch-shifting process was implemented using the sox implementation of the WSOLA algorithm in the [pysox](https://github.com/rabitt/pysox) package. We created modified the utterances in the selected dataset by shifting 1 and 2 semitones (i.e. 100 and 200 cents, respectively) up and down. Thus, for each audio file, there are 4 pitch-shifted versions.
# 
# Finally, the naming convention for the audio files is: *{ID}_{File Number}_{XX}_{Label}.wav* (e.g. p236_002_mic1_1semitonedown.wav).

# ## 1) Loading Data

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import deciphering_enigma
import matplotlib.pyplot as plt

#define the experiment config file path
path_to_config = './config.yaml'

#read the experiment config file
exp_config = deciphering_enigma.load_yaml_config(path_to_config)

#register experiment directory and read wav files' paths
audio_files = deciphering_enigma.build_experiment(exp_config)
audio_files = [s for s in audio_files if s.endswith(f'up.wav') or s.endswith('1_normloud.wav')]
print(f'Dataset has {len(audio_files)} samples')


# In[ ]:


#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
metadata_df['Label'] = metadata_df['Label'].str.replace('wav','orig')
metadata_df.drop(columns=['xx'], inplace=True)

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(audio_files, cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format)


# ## 2) Generating Embeddings

# We are generating speech embeddings from 15 different models.

# In[4]:


#generate speech embeddings
feature_extractor = deciphering_enigma.FeatureExtractor()
embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)


# In[3]:


import matplotlib
from pylab import cm
import matplotlib as mpl
matplotlib.font_manager._fmcache
matplotlib.font_manager._rebuild()
mpl.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3


# ## 3) Linear Encoding

# In this section, we evaluate the predictive power of all models to recognize speakers after shifting their pitch. We used different ML classifiers such as Logistic Regression (LR), Random Forest (RF), and Support Vector Machine (SVM). The work flow of this section, for each model, is as follows:
# 1) Split the dataset into train and test sets (70% and 30%, respectively) (Note: the splitting here is done per speaker to maintain equal ratio of samples across speakers)
# 2) After extracting the representations of both splits from all models, the classifier will be only trained on the representations/embeddings of the original utterances (no pitch-shifting) in the train set.
# 3) Then tested on all conditions in the test set (i.e. original, 1 semitone Up, 2 semitone Up,...etc.).
# 4) For each classifier, we implement grid search across its hyperparameters and we report performance for each hyperparameters sequence on the test set
# 5) We used Predefined splits for train and test.
# 6) The pipeline comprises standardizing the representations then passing it to the classifier.
# 7) Consequently, the encoder yields several UAR scores for each classifier, each label and eventually each model. (illustrated in the box plots below)

# In[12]:


ml_encoder = ML_Encoder()
for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
    ml_encoder.run(model_name, embeddings, metadata_df['Label'], metadata_df['ID'], exp_config.dataset_name)


# In[21]:


#read scores from all models across all classifiers as dataframe
df = pd.read_csv(f'../{exp_config.dataset_name}/linear_encoding_all_models_scores.csv', index_col=0)


# In[25]:


order = ['2 Semitone Down', '1 Semitone Down', 'Original', '1 Semitone Up', '2 Semitone Up']


# ### 3.1. Logistic Regression

# In[27]:


clf = 'LR'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=order, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ### 3.2. Random Forest

# In[28]:


clf = 'RF'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=order, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ### 3.3. Support Vector Machine

# In[29]:


clf = 'SVM'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=order, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ## 4) Dimensionality Reduction

# The previous analysis showed how well the model is capable of grouping the uttereances of the same speaker in different cases (scripted and spontaneous) in the embedding space (high dimension). That being said, we will replicate the same analysis but in the lower dimension space to visualize the impact of speaking styles on voice identity perception.

# Accordingly, we will utilize different kind of dimensionality reduction such as PCA, tSNE, UMAP and PaCMAP to get a better idea of how the speakers' samples are clustered together in 2D. However, one constraint is that these methods are sensitive to their hyperparameters (except PCA) which could imapct our interpretation of the results. Thus, a grid search across the hyperparameters for each method is implemented.

# Another issue would be quantifying the ability of these methods to perserve the distances amongst samples in the high dimension and present it in a lower dimension. To address this, we are using two metrics KNN and CPD that represent the ability of the algorithm to preserve local and global structures of the original embedding space, respectively. Both metrics are adopted from this [paper](https://www.nature.com/articles/s41467-019-13056-x) in which they define both metrics as follows:
# 
# * KNN: The fraction of k-nearest neighbours in the original high-dimensional data that are preserved as k-nearest neighbours in the embedding. KNN quantifies preservation of the local, or microscopic structure. The value of K used here is the min number of samples a speaker would have in the original space.
# 
# * CPD: Spearman correlation between pairwise distances in the high-dimensional space and in the reduced embedding. CPD quantifies preservation of the global, or macroscropic structure. Computed across all pairs among 1000 randomly chosen points without replacement. 

# Consequently, we present the results from dimensionality reduction methods in two ways, one optimimizing local structure metric (KNN) and the other optimizing global structure metric (CPD).

# In[6]:


tuner = deciphering_enigma.ReducerTuner()
for i, model_name in enumerate(embeddings_dict.keys()):
    print(f'{model_name}:')
    tuner.tune_reducer(embeddings_dict[model_name], metadata=metadata_df, dataset_name=exp_config.dataset_name, model_name=model_name)


# In[31]:


handcrafted_features = ['Log-Mel-Spectrogram', 'Cochleagram']
byol_models = ['BYOL-A_default', 'BYOL-S_default']
cvt_models = ['BYOL-S_cvt', 'Hybrid_BYOL-S_cvt']
generative_models = ['TERA', 'APC']
wav2vec2_models = ['Wav2Vec2_latent', 'Wav2Vec2']
hubert_models = ['HuBERT_latent', 'HuBERT']
data2vec_models = ['Data2Vec_latent', 'Data2Vec']


# In[73]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', red_name='PCA'):
    plot = sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name, palette='deep', style=label_name, ax=axis)
    axis.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axis.tick_params(left=False, bottom=False)
    axis.get_legend().remove()


# ### <a id='another_cell'></a> 4.1 Pitch-shifting Up

# #### 4.1.1. Mapping Gender

# In[33]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[34]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[35]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[36]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[37]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[38]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[39]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.1.2. Mapping Identity

# In[40]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[41]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[42]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[43]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[44]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[45]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[46]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.1.3. Mapping Shift in Pitch

# In[97]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[98]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[99]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[100]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[101]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[102]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[103]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# ### <a id='another_cell'></a> 4.2 Pitch-shifting Down

# #### 4.2.1. Mapping Gender

# In[75]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[76]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[77]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[78]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[79]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[80]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[81]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Gender'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.2.2. Mapping Identity

# In[82]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[83]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[84]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[85]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[86]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[87]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[88]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'ID'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.2.3. Mapping Shift in Pitch

# In[89]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[90]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[91]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[92]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[93]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[94]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[95]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')

