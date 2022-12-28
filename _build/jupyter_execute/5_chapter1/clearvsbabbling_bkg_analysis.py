#!/usr/bin/env python
# coding: utf-8

# # Clear vs. Babbling Background Experiment

# In this section, we explore the impact of the audio background on voice identity processing.

# ### Research Questions:
# 1) Which models would be invariant to background noise?
# 2) How could background noise impact the models ability to predict identity?

# ## Dataset Description:

# The dataset used in this experiment is [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443). We specifically used a [noisy version](https://datashare.ed.ac.uk/handle/10283/2791) of the dataset. This version comprises clean and noisy (babbling sound) parallel speech database. It comprises speech utterances from 110 english speakers with different accents. However, we created a mini version of this dataset by selecting only 10 speakers (5 Males / 5 Females). Each speaker reads about 400 scentences.

# The dataset was processed by undersampling to 16 kHz to be compatible with the pre-trained self-supervised models.

# ## 1) Loading Data

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import deciphering_enigma
import matplotlib.pyplot as plt

#define the experiment config file path
path_to_config = './config.yaml'

#read the experiment config file
exp_config = deciphering_enigma.load_yaml_config(path_to_config)
dataset_path = exp_config.dataset_path

#register experiment directory and read wav files' paths
audio_files = deciphering_enigma.build_experiment(exp_config)
print(f'Dataset has {len(audio_files)} samples')


# In[5]:


if exp_config.preprocess_data:
    dataset_path = deciphering_enigma.preprocess_audio_files(audio_files, speaker_ids=metadata_df['ID'], chunk_dur=exp_config.chunk_dur, resampling_rate=exp_config.resampling_rate, 
                    save_path=f'{exp_config.dataset_name}_{exp_config.model_name}/preprocessed_audios', audio_format=audio_format)

#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
metadata_df['Label'] = np.array(list(map(lambda x: x.split('/')[-3], audio_files)))
metadata_df['Gender'] = np.array(list(map(lambda x: x.split('/')[-2][-1], audio_files)))
metadata_df.drop(columns=['xx'], inplace=True)

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(audio_files, cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format)


# ## 2) Generating Embeddings

# We are generating speech embeddings from 15 models.

# In[6]:


#generate speech embeddings
feature_extractor = deciphering_enigma.FeatureExtractor()
embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)


# In[7]:


import matplotlib
from pylab import cm
import matplotlib as mpl
matplotlib.font_manager._fmcache
matplotlib.font_manager._rebuild()
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3


# ## 3) Linear Encoding:

# In this section, we evaluate the predictive power of all models to recognize speakers after shifting their pitch. We used different ML classifiers such as Logistic Regression (LR), Random Forest (RF), and Support Vector Machine (SVM). The work flow of this section, for each model, is as follows:
# 1) Split the dataset into train and test sets (70% and 30%, respectively) (Note: the splitting here is done per speaker to maintain equal ratio of samples across speakers)
# 2) After extracting the representations of both splits from all models, the classifier will be only trained on the representations/embeddings of the original utterances (clear background) in the train set.
# 3) Then tested on all conditions in the test set (i.e. clear and babbling backgrounds).
# 4) For each classifier, we implement grid search across its hyperparameters and we report performance for each hyperparameters sequence on the test set
# 5) We used Predefined splits for train and test.
# 6) The pipeline comprises standardizing the representations then passing it to the classifier.
# 7) Consequently, the encoder yields several UAR scores for each classifier, each label and eventually each model. (illustrated in the box plots below)

# In[7]:


ml_encoder = deciphering_enigma.ML_Encoder()
for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
    ml_encoder.run(model_name, embeddings, metadata_df['Label'], metadata_df['ID'], exp_config.dataset_name, 'clear')


# In[8]:


#read scores from all models across all classifiers as dataframe
df = pd.read_csv(f'../{exp_config.dataset_name}/linear_encoding_all_models_scores.csv', index_col=0)


# In[14]:


palette = {'Clear Bkg.':'cadetblue', 'Babbling Bkg.':'firebrick'}


# ### 3.1. Logistic Regression

# In[23]:


clf = 'LR'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=['Clear Bkg.', 'Babbling Bkg.'], ax=ax, palette=palette)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.axhline(y=10, color='orange')
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ### 3.2. Random Forest

# In[24]:


clf = 'RF'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=['Clear Bkg.', 'Babbling Bkg.'], ax=ax, palette=palette)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.axhline(y=10, color='orange')
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ### 3.3. Support Vector Machine

# In[25]:


clf = 'SVM'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', hue_order=['Clear Bkg.', 'Babbling Bkg.'], ax=ax, palette=palette)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.axhline(y=10, color='orange')
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}.svg')


# ## 4) Similarity Representation Analysis:

# In[26]:


cka_class = deciphering_enigma.CKA(unbiased=True, kernel='rbf', rbf_threshold=0.5)
cka_ = cka_class.run_cka(embeddings_dict, compute_from='features', save_path=f'../{exp_config.dataset_name}/', save_array=True)
cka_class.plot_heatmap(cka_, embeddings_dict.keys(), save_path=f'../{exp_config.dataset_name}', save_fig=True)


# ## 5) Dimensionality Reduction

# The previous analysis showed how well the model is capable of grouping the uttereances of the same speaker in different cases (scripted and spontaneous) in the embedding space (high dimension). That being said, we will replicate the same analysis but in the lower dimension space to visualize the impact of speaking styles on voice identity perception.

# Accordingly, we will utilize different kind of dimensionality reduction such as PCA, tSNE, UMAP and PaCMAP to get a better idea of how the speakers' samples are clustered together in 2D. However, one constraint is that these methods are sensitive to their hyperparameters (except PCA) which could imapct our interpretation of the results. Thus, a grid search across the hyperparameters for each method is implemented.

# Another issue would be quantifying the ability of these methods to perserve the distances amongst samples in the high dimension and present it in a lower dimension. To address this, we are using two metrics KNN and CPD that represent the ability of the algorithm to preserve local and global structures of the original embedding space, respectively. Both metrics are adopted from this [paper](https://www.nature.com/articles/s41467-019-13056-x) in which they define both metrics as follows:
# 
# * KNN: The fraction of k-nearest neighbours in the original high-dimensional data that are preserved as k-nearest neighbours in the embedding. KNN quantifies preservation of the local, or microscopic structure. The value of K used here is the min number of samples a speaker would have in the original space.
# 
# * CPD: Spearman correlation between pairwise distances in the high-dimensional space and in the reduced embedding. CPD quantifies preservation of the global, or macroscropic structure. Computed across all pairs among 1000 randomly chosen points without replacement. 

# Consequently, we present the results from dimensionality reduction methods in two ways, one optimimizing local structure metric (KNN) and the other optimizing global structure metric (CPD).

# In[27]:


tuner = deciphering_enigma.ReducerTuner()
for i, model_name in enumerate(embeddings_dict.keys()):
    print(f'{model_name}:')
    tuner.tune_reducer(embeddings_dict[model_name], metadata=metadata_df, dataset_name=exp_config.dataset_name, model_name=model_name)


# ### <a id='another_cell'></a> 5.1 Mapping Labels

# In[28]:


handcrafted_features = ['Log-Mel-Spectrogram', 'Cochleagram']
byol_models = ['BYOL-A_default', 'BYOL-S_default']
cvt_models = ['BYOL-S_cvt', 'Hybrid_BYOL-S_cvt']
generative_models = ['TERA', 'APC']
wav2vec2_models = ['Wav2Vec2_latent', 'Wav2Vec2']
hubert_models = ['HuBERT_latent', 'HuBERT']
data2vec_models = ['Data2Vec_latent', 'Data2Vec']


# In[29]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', red_name='PCA'):
    plot = sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name, palette='deep', style=label_name, ax=axis)
    axis.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axis.tick_params(left=False, bottom=False)
    axis.get_legend().remove()


# #### 5.1.1. Mapping Gender

# In[30]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[35]:


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


# In[36]:


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


# #### 5.1.2. Mapping Identity

# In[37]:


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


# In[38]:


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


# In[39]:


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


# In[40]:


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


# In[41]:


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


# In[42]:


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


# In[43]:


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


# #### 4.1.3. Mapping Background label (Clear/Babbling)

# In[44]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[45]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[46]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[47]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
ax[0,j].legend(bbox_to_anchor=(1, 1), fontsize=30)
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[48]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
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


# In[49]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
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


# In[50]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Label'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df['Label'] = df['Label'].str.replace('orig', 'Forward')
    df['Label'] = df['Label'].str.replace('reverse', 'Backward')
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


# ### 5.2 Mapping Acoustic Features

# In[52]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', red_name='PCA'):
    points = axis.scatter(df[red_name, opt_structure, 'Dim1'], df[red_name, opt_structure, 'Dim2'],
                     c=df[label_name], s=10, cmap="Spectral")
    axis.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axis.tick_params(left=False, bottom=False)
    return points


# #### 5.2.1. Fundamental Frequency

# In[39]:


f0s = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='f0')


# In[53]:


f0s = np.load(f'../{exp_config.dataset_name}/f0.npy')


# In[54]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[55]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[56]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[57]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[58]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[59]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[60]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'f0'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(f0s)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median F0', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.2.2. RMS

# In[49]:


rms = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='rms')


# In[61]:


rms = np.load(f'../{exp_config.dataset_name}/rms.npy')
rms = np.log(rms)


# In[62]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[63]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[64]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[65]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[66]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[67]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[68]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'rms'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = rms
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log Median RMS', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.2.3. First MFCC

# In[71]:


mfcc_start = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='mfcc', mfcc_num=1)


# In[69]:


mfcc_start = np.load(f'../{exp_config.dataset_name}/mfcc.npy')


# In[70]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[71]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[72]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[73]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[74]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[75]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[76]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'mfcc_start'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = mfcc_start
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('First MFCC', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')


# #### 4.2.4. Number of Syllables

# In[72]:


num_syl = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='num_syl')


# In[77]:


num_syl = np.load(f'../{exp_config.dataset_name}/num_syl.npy')


# In[78]:


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.histplot(x=num_syl, kde=True, ax=ax, bins=15)
ax.set_xlabel('# of Syllables')
plt.savefig(f'../{exp_config.dataset_name}/num_syl.png')


# In[79]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[80]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[81]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[82]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[83]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[84]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[85]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = np.log(num_syl)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('Log(# of Syllables)', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')

