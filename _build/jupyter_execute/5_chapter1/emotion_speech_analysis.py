#!/usr/bin/env python
# coding: utf-8

# # Emotion Speech Experiment

# In this section, we explore the impact of emotions on voice identity processing.

# ### Research Questions:
# 1) How do different models perceive changes in emotions in speech?
# 2) Which models would be invariant to emotions?
# 3) How could variation in paralinguistic input (e.g. emotions) impact the models ability to predict identity?

# ## Dataset Description:

# In this experiment, we used a dataset called *[SUBESCO](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250173)*. It is a speech emotion recognition (SER) dataset in Bangla language. The corpus contains 7 hours of speech. It comprises 7000 utterances from 20 native speakers (10 Females/10 Males) expressing 7 different emotions (Neutral, Happy, Sad, Fear, Angry, Surprise, and Disgust). Each speaker produced 10 scentences with 7 emotions and repeated 5 times. Hence, $20*10*7*5 = 7000$ utterances.

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
dataset_path = exp_config.dataset_path

#register experiment directory and read wav files' paths
audio_files = deciphering_enigma.build_experiment(exp_config)
print(f'Dataset has {len(audio_files)} samples')


# In[3]:


#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
metadata_df.drop(columns=['xx'], inplace=True)
metadata_df['Path'] = audio_files

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(metadata_df.Path, cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format)


# ## 2) Generating Embeddings

# We are generating speech embeddings from 15 different models.

# In[4]:


#generate speech embeddings
feature_extractor = deciphering_enigma.FeatureExtractor()
embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)


# In[5]:


import matplotlib
from pylab import cm
import matplotlib as mpl
matplotlib.font_manager._fmcache
matplotlib.font_manager._rebuild()
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 3


# ## 3) Linear Encoding:

# In this section, we evaluate the self-supervised models ability to identify speakers from different emotions by leveraging different ML classifiers such as Logistic Regression (LR), Random Forest (RF) and Support Vector Machine (SVM). First, for each model, we trained a classifier on one emotion and tested on the same but unseen scentences. The results for this experiment is shown under *Training and Testing on the same emotion*. This experiment highlights the impact of emotions on speaker recognition. Also, it shows the most robust emotion for each model for such task. The other experiments are basically training the classifiers on one emotion (Neutral/Angry) and testing on all emotions (unseen scentences). This experiment might highlight the robustness of the models to identify speakers in different paralinguistic states.

# In[ ]:


ml_encoder = deciphering_enigma.ML_Encoder()
for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
    ml_encoder.run(model_name, embeddings, metadata_df['Label'], metadata_df['ID'], metadata_df['Scentence'], exp_config.dataset_name)


# In[17]:


emotions_order = ['NEUTRAL', 'SAD', 'DISGUST', 'FEAR', 'HAPPY', 'SURPRISE', 'ANGRY']


# ### 3.1. Training and Testing on the same emotion

# In[16]:


#read scores from all models across all classifiers as dataframe
df = pd.read_csv(f'../{exp_config.dataset_name}/linear_encoding_scores_allmodels_peremotion.csv', index_col=0)


# #### 3.1.1 Logistic Regression

# In[18]:


clf = 'LR'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.svg')


# #### 3.1.2. Random Forest

# In[19]:


clf = 'RF'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.svg')


# #### 3.1.3. Support Vector Machine

# In[20]:


clf = 'SVM'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.svg')


# #### 3.1.4. Shallow Layer

# In[21]:


clf = 'MLP'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_peremotion.svg')


# ### 3.2. Training on Neutral and Testing on the all emotions

# In[35]:


#read scores from all models across all classifiers as dataframe
df = pd.read_csv(f'../{exp_config.dataset_name}/linear_encoding_scores_allmodels_label_neutral.csv', index_col=0)


# In[41]:


emotions_order = ['NEUTRAL', 'SAD', 'FEAR', 'DISGUST', 'HAPPY', 'SURPRISE', 'ANGRY']


# #### 3.2.1 Logistic Regression

# In[42]:


clf = 'LR'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.svg')


# #### 3.2.2. Random Forest

# In[43]:


clf = 'RF'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.svg')


# #### 3.2.3. Support Vector Machine

# In[44]:


clf = 'SVM'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.svg')


# #### 3.2.4. Shallow Layer

# In[45]:


clf = 'MLP'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_neutral.svg')


# ### 3.3. Training on Angry and Testing on the all emotions

# In[46]:


df = pd.DataFrame({'Model': [], 'Label': [], 'Clf': [], 'Score': []})
for model_name in embeddings_dict.keys():
    df_model = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/linear_encoding_scores_label_angry.csv')
    df_model['Score'] = df_model['Score']*100
    df = pd.concat([df, df_model])
df.to_csv(f'../{exp_config.dataset_name}/linear_encoding_scores_allmodels_label_angry.csv')


# In[47]:


#read scores from all models across all classifiers as dataframe
df = pd.read_csv(f'../{exp_config.dataset_name}/linear_encoding_scores_allmodels_label_angry.csv', index_col=0)


# In[48]:


emotions_order = ['NEUTRAL', 'SAD', 'FEAR', 'DISGUST', 'HAPPY', 'SURPRISE', 'ANGRY']


# #### 3.3.1 Logistic Regression

# In[49]:


clf = 'LR'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.svg')


# #### 3.3.2. Random Forest

# In[50]:


clf = 'RF'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.svg')


# #### 3.3.3. Support Vector Machine

# In[51]:


clf = 'SVM'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.svg')


# #### 3.3.4. Shallow Layer

# In[52]:


clf = 'MLP'
fig, ax = plt.subplots(1, 1, figsize=(25, 15))
graph = sns.barplot(data=df.loc[df.Clf == clf], x='Model', y='Score', hue='Label', ax=ax, hue_order=emotions_order)
ax.set_xticklabels(ax.get_xticklabels(), size = 35, rotation=80)
ax.set_yticklabels(ax.get_yticks(), size = 35)
ax.set_ylabel('UAR (%)', fontsize=40)
ax.set_xlabel('Models', fontsize=40)
ax.legend(bbox_to_anchor=(1, 1), fontsize=40)
plt.grid()
plt.tight_layout()
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.png')
plt.savefig(f'../{exp_config.dataset_name}/linear_encoding_plots/linear_encoding_{clf}_angry.svg')


# ## 4) Similarity Representation Analysis:

# In[53]:


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
# * CPD: Spearman correlation between pairwise distances in the high-dimensional space and in the embedding. CPD quantifies preservation of the global, or macroscropic structure. Computed across all pairs among 1000 randomly chosen points with replacement. 

# Consequently, we present the results from dimensionality reduction methods in two ways, one optimimizing local structure metric (KNN) and the other optimizing global structure metric (CPD).

# In[22]:


tuner = deciphering_enigma.ReducerTuner()
for i, model_name in enumerate(embeddings_dict.keys()):
    print(f'{model_name}:')
    tuner.tune_reducer(embeddings_dict[model_name], metadata=metadata_df, dataset_name=exp_config.dataset_name, model_name=model_name)


# ### <a id='another_cell'></a> 5.1 Mapping Labels

# In[6]:


handcrafted_features = ['Log-Mel-Spectrogram', 'Cochleagram']
byol_models = ['BYOL-A_default', 'BYOL-S_default', 'BYOL-I_default']
cvt_models = ['BYOL-S_cvt', 'Hybrid_BYOL-S_cvt']
generative_models = ['TERA', 'APC']
wav2vec2_models = ['Wav2Vec2_latent', 'Wav2Vec2']
hubert_models = ['HuBERT_latent', 'HuBERT']
data2vec_models = ['Data2Vec_latent', 'Data2Vec']


# In[24]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', red_name='PCA'):
    plot = sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name, palette='deep', style=label_name, ax=axis)
    axis.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axis.tick_params(left=False, bottom=False)
    axis.get_legend().remove()


# #### 5.1.1. Mapping Gender

# In[26]:


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


# In[27]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[28]:


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


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


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

# In[33]:


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


# In[34]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[35]:


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


# In[36]:


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


# In[37]:


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


# In[38]:


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


# In[39]:


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


# #### 4.1.3. Mapping Emotions

# In[40]:


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


# In[41]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[42]:


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


# In[43]:


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


# In[44]:


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


# In[45]:


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


# In[46]:


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


# #### 4.1.4. Mapping Scentences

# In[49]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# In[50]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
optimize = 'Global'
label = 'Scentence'
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


# In[51]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# In[52]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# In[54]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# In[55]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# In[56]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'Scentence'
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


# ### 5.2 Mapping Acoustic Features

# In[10]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', red_name='PCA'):
    points = axis.scatter(df[red_name, opt_structure, 'Dim1'], df[red_name, opt_structure, 'Dim2'],
                     c=df[label_name], s=10, cmap="Spectral")
    axis.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axis.tick_params(left=False, bottom=False)
    return points


# #### 5.2.1. Fundamental Frequency

# In[ ]:


f0s = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='f0')


# In[11]:


f0s = np.load(f'../{exp_config.dataset_name}/f0.npy')


# In[12]:


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


# In[13]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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

# In[ ]:


rms = deciphering_enigma.compute_acoustic_features(audio_files, save_path=f'../{exp_config.dataset_name}', feature='rms')


# In[19]:


rms = np.load(f'../{exp_config.dataset_name}/rms.npy')
rms = np.log(rms)


# In[20]:


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


# In[21]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


mfcc_start = np.load(f'../{exp_config.dataset_name}/mfcc.npy')


# In[28]:


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


# In[29]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
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


# In[30]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[35]:


num_syl = np.load(f'../{exp_config.dataset_name}/num_syl.npy')


# In[36]:


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.histplot(x=num_syl, kde=True, ax=ax, bins=15)
ax.set_xlabel('# of Syllables')
plt.savefig(f'../{exp_config.dataset_name}/num_syl.png')


# In[40]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(handcrafted_features):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/handcrafted_{label}_dimred.svg')


# In[42]:


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(byol_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/byol_{label}_dimred.svg')


# In[43]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(cvt_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/cvt_{label}_dimred.svg')


# In[44]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(generative_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/generative_{label}_dimred.svg')


# In[70]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(wav2vec2_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Wav2Vec2':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/wav2vec2_{label}_dimred.svg')


# In[71]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(hubert_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'HuBERT':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/hubert_{label}_dimred.svg')


# In[72]:


fig, ax = plt.subplots(2, 4, figsize=(20, 10))
optimize = 'Global'
label = 'num_syl'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(data2vec_models):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    df[label] = num_syl
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=30)
        points = visualize_embeddings(df, label, metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name)
    if model_name == 'Data2Vec':
        model_name += '_final'
    ax[i, 0].set_ylabel(model_name, fontsize=30)
cbar = fig.colorbar(points, ax=ax, pad=0.02)
cbar.set_label('# of Syllables', rotation=270, labelpad=25, fontsize=30)
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.png')
plt.savefig(f'../{exp_config.dataset_name}/dim_red_plots/data2vec_{label}_dimred.svg')

