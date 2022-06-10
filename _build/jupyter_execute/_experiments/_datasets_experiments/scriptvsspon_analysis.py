#!/usr/bin/env python
# coding: utf-8

# # Scripted vs Spontaneous Speech

# In this section, we explore the differences between scripted speech and casual/spontaneous speech. Both speaking styles feature minimal vocal variations yet impactful. It has been observed that speaking style could affect voice perception in humans in case of unfamiliar voices ([Smith et al. (2019)](https://onlinelibrary.wiley.com/doi/epdf/10.1002/acp.3478?saml_referrer), [Stevenage et al. (2021)](https://link.springer.com/content/pdf/10.1007/s10919-020-00348-w.pdf) and [Afshan et al. (2022)](https://asa.scitation.org/doi/pdf/10.1121/10.0009585?casa_token=rSyTJ-uiRW8AAAAA:GlyYCKNccGdLEfnk5oynoj-IgLnAlSBPuHTndx8uzg0VsupZ3bOFqfGJROBRhBxdcgs6ozZR0DvL)). Accordingly, we are going to investigate the effect of speaking style on generating speech embeddings that should maintain close distances with samples from the same speaker.

# ### <span style="color:blue"><div style="text-align: justify">Research Questions:</div></span>
# 1) Is there a noticeable within-speaker difference between scripted and spontaneous speech utterances?
# 2) Would the difference change depending on the type of feature extrator used?
# 3) Is this difference maintained in lower dimensions?

# ## Dataset Description:

# The dataset used in this experiment is obtained from [here](https://speechbox.linguistics.northwestern.edu/ALLSSTARcentral/#!/recordings). We compiled speech utterances from 26 speakers (14 females and 12 males). The collected dataset comprises 7 tasks (4 scripted/3 spontaneous).
# 
# Tasks:
# 
# 1) NWS (script): Reading *'The North Wind and Sun'* passage
# 2) LPP (script): Reading *'The Little Prince'* scentences
# 3) DHR (script): Reading *'Declaration of Human Rights'* scentences
# 4) HT2 (script): Reading *'Hearing in Noise Test 2'* scentences
# 5) QNA (spon): Answering questions *'Q and A session'*
# 6) ST1 (spon): Telling a personal story 1
# 7) ST2 (spon): Telling a personal story 2

# The dataset was processed by undersampling to 16 kHz to be compatible with BYOL-S models. Additionally, the utterances were cropped to fixed durations (1, 3, 5, 10, 15 sec) to yield 5 new datasets generated from the original one.
# 
# Finally, the naming convention for the audio files is: *{ID}_{Gender}_{Task}_{Label}_{File Number}.wav* (e.g. 049_F_DHR_script_000.wav).

# In the following analysis, we will be using the 3sec-utterance version of the dataset.

# ## 1) Loading Data

# In[4]:


import deciphering_enigma

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
#balance data to have equal number of labels per speaker
audio_files = deciphering_enigma.balance_data()
print(f'After Balancing labels: Dataset has {len(audio_files)} samples')

#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(audio_files, cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format)


# ## 2) Generating Embeddings

# We are generating speech embeddings from 9 different models (BYOL-A, BYOL-S/CNN, BYOL-S/CvT, Hybrid BYOL-S/CNN, Hybrid BYOL-S/CvT, Wav2Vec2, HuBERT and Data2Vec).

# In[6]:


#generate speech embeddings
embeddings_dict = deciphering_enigma.extract_models(audio_tensor_list, exp_config)


# ## 3) Original Dimension Analysis

# ### 3.1. Distance-based

# Compute distances (e.g. cosine distance) across embeddings of utterances. Steps to compute it:
# 
# 1) Compute distances across all 5816 samples in a pairwise format (5816\*5816).
# 2) Convert pairwise form to long form i.e. two long columns [Sample1, Sample2, Distance], yielding a dataframe of 5816\*5816 long.
# 3) Remove rows with zero distances (i.e. distances between a sample and itself).
# 4) Keep only the distances between samples from the same speaker and the same label (e.g. Dist{speaker1_Label1_audio0 --> speaker1_Label1_audio1}), as shown in figure below.
# 5) Remove duplicates, i.e. distance between 0 --> 1 == 1 --> 0.
# 6) Standardize distances within each speaker to account for within speaker variability space.
# 7) Remove the distances above 99% percentile (outliers).
# 8) Plot violin plot for each model, split by the label to see how are these models encode both labels.

# ![distance](distance_orig_dimensions.png)

# In[5]:


df_all = deciphering_enigma.compute_distances(metadata_df, embeddings_dict, exp_config.dataset_name, 'cosine', list(metadata_df.columns))


# In[6]:


deciphering_enigma.visualize_violin_dist(df_all)


# ### 3.2. Similarity Representation Analysis:

# In[11]:


import numpy as np
from tqdm import tqdm
cka_class = deciphering_enigma.CKA(unbiased=True, kernel='rbf', rbf_threshold=0.5)
num_models = len(embeddings_dict.keys())
cka_ = np.zeros((num_models, num_models))
print(cka_.shape)
for i, (_, model_1) in enumerate(tqdm(embeddings_dict.items())):
    for j, (_, model_2) in enumerate(embeddings_dict.items()):
        cka_[i,j] = cka_class.compute(model_1, model_2)


# In[12]:


cka_class.plot_heatmap(cka_, embeddings_dict.keys(), save_path=f'{exp_config.dataset_name}', save_fig=True)


# ## 4) Dimensionality Reduction

# The previous analysis showed how well the model is capable of grouping the uttereances of the same speaker in different cases (scripted and spontaneous) in the embedding space (high dimension). That being said, we will replicate the same analysis but in the lower dimension space to visualize the impact of speaking styles on voice identity perception.

# Accordingly, we will utilize different kind of dimensionality reduction such as PCA, tSNE, UMAP and PaCMAP to get a better idea of how the speakers' samples are clustered together in 2D. However, one constraint is that these methods are sensitive to their hyperparameters (except PCA) which could imapct our interpretation of the results. Thus, a grid search across the hyperparameters for each method is implemented.

# Another issue would be quantifying the ability of these methods to perserve the distances amongst samples in the high dimension and present it in a lower dimension. To address this, we are using two metrics KNN and CPD that represent the ability of the algorithm to preserve local and global structures of the original embedding space, respectively. Both metrics are adopted from this [paper](https://www.nature.com/articles/s41467-019-13056-x) in which they define both metrics as follows:
# 
# * KNN: The fraction of k-nearest neighbours in the original high-dimensional data that are preserved as k-nearest neighbours in the embedding. KNN quantifies preservation of the local, or microscopic structure. The value of K used here is the min number of samples a speaker would have in the original space.
# 
# * CPD: Spearman correlation between pairwise distances in the high-dimensional space and in the embedding. CPD quantifies preservation of the global, or macroscropic structure. Computed across all pairs among 1000 randomly chosen points with replacement. 

# Consequently, we present the results from dimensionality reduction methods in two ways, one optimimizing local structure metric (KNN) and the other optimizing global structure metric (CPD).

# ### <a id='another_cell'></a> 4.1 Mapping Labels

# In[7]:


tuner = deciphering_enigma.ReducerTuner()
for i, model_name in enumerate(embeddings_dict.keys()):
    tuner.tune_reducer(embeddings_dict[model_name], metadata=metadata_df, dataset_name=exp_config.dataset_name, model_name=model_name)


# In[20]:


import seaborn as sns
def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', plot_type='sns', red_name='PCA', row=1, col=1, hovertext='', label='spon'):
    if plot_type == 'sns':
        if label_name == 'Gender':
            sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name, palette='deep', ax=axis)
        else:
            sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name
                            , style=label_name, palette='deep', ax=axis)
        axis.set(xlabel=None, ylabel=None)
        axis.get_legend().remove()
    elif plot_type == 'plotly':
        traces = px.scatter(x=df[red_name, opt_structure, 'Dim1'], y=df[red_name, opt_structure, 'Dim2'], color=df[label_name].astype(str), hover_name=hovertext)
        traces.layout.update(showlegend=False)
        axis.add_traces(
            list(traces.select_traces()),
            rows=row, cols=col
        )
    else:
        points = axis.scatter(df[red_name, opt_structure, 'Dim1'], df[red_name, opt_structure, 'Dim2'],
                     c=df[label_name], s=20, cmap="Spectral")
        return points


# #### 4.1.1. Mapping Gender

# In[22]:


import matplotlib.pyplot as plt
import pandas as pd
fig, ax = plt.subplots(9, 4, figsize=(40, 90))
optimize = 'Global'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(embeddings_dict.keys()):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=25)
        visualize_embeddings(df, 'Gender', metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='sns')
    ax[i, 0].set_ylabel(model_name, fontsize=25)
ax[0,j].legend(bbox_to_anchor=(1, 1.15), fontsize=20)
plt.tight_layout()


# #### 4.1.2. Mapping Identity

# In[26]:


fig, ax = plt.subplots(9, 4, figsize=(40, 90))
optimize = 'Global'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(embeddings_dict.keys()):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=25)
        visualize_embeddings(df, 'ID', metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='sns')
    ax[i, 0].set_ylabel(model_name, fontsize=25)
plt.tight_layout()


# #### 4.1.3. Mapping Speaking Style (Script/Spon)

# In[33]:


fig, ax = plt.subplots(9, 4, figsize=(40, 90))
optimize = 'Global'
reducer_names = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
for i, model_name in enumerate(embeddings_dict.keys()):
    df = pd.read_csv(f'../{exp_config.dataset_name}/{model_name}/dim_reduction.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': '', 'Unnamed: 20_level_1': '', 'Unnamed: 20_level_2': '',
                       'Unnamed: 21_level_1': '', 'Unnamed: 21_level_2': '',},inplace=True)
    for j, name in enumerate(reducer_names):
        ax[0,j].set_title(f'{name}', fontsize=25)
        visualize_embeddings(df, 'Label', metrics=[], axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='sns')
    ax[i, 0].set_ylabel(model_name, fontsize=25)
ax[0,j].legend(bbox_to_anchor=(1, 1.15), fontsize=20)
plt.tight_layout()


# ### 4.2 Distance in Lower Dimensions

# In[217]:


labels = ['script', 'spon']
dfs = []
for label in labels:
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    pacmap_global_df = df.loc[:, ('PaCMAP', 'Global')]
    pacmap_global_df['wav_file'] = df['wav_file']; pacmap_global_df['label'] = label
    dfs.append(pacmap_global_df)
df = pd.concat(dfs, axis=0)
df.sample(10)


# In[219]:


#create distance-based dataframe between all data samples in a square form
pairwise = pd.DataFrame(
    squareform(pdist(df.iloc[:, :2], metric='cosine')),
    columns = df['wav_file'],
    index = df['wav_file']
)


# In[221]:


#move from square form DF to long form DF
long_form = pairwise.unstack()
#rename columns and turn into a dataframe
long_form.index.rename(['Sample_1', 'Sample_2'], inplace=True)
long_form = long_form.to_frame('Distance').reset_index()
#remove the distances computed between same samples (distance = 0)
long_form = long_form.loc[long_form['Sample_1'] != long_form['Sample_2']]
long_form.sample(10)


# In[223]:


#add columns for meta-data
long_form['Gender'] = long_form.apply(lambda row: row['Sample_1'].split('_')[1] if row['Sample_1'].split('_')[1] == row['Sample_2'].split('_')[1] else 'Different', axis=1)
long_form['Label'] = long_form.apply(lambda row: row['Sample_1'].split('_')[3] if row['Sample_1'].split('_')[3] == row['Sample_2'].split('_')[3] else 'Different', axis=1)
long_form['ID'] = long_form.apply(lambda row: row['Sample_1'].split('_')[0] if row['Sample_1'].split('_')[0] == row['Sample_2'].split('_')[0] else 'Different', axis=1)
long_form.sample(10)


# In[224]:


#remove distances computed between different speakers and different labels
df = long_form.loc[(long_form['Gender']!='Different') & (long_form['Label']!='Different') & (long_form['ID']!='Different')]
df.sample(10)


# In[226]:


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.violinplot(data=df, x='Label', y='Distance', inner='quartile', ax=ax)
ax.set_xlabel('Labels', fontsize=15)
ax.set_ylabel('Cosine Distances', fontsize=15)

# statistical annotation
d=cohend(df['Distance'].loc[(df.Label=='spon')], df['Distance'].loc[(df.Label=='script')])
x1, x2 = 0, 1
y, h, col = df['Distance'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+(h*1.5), f'cohen d={d:.2}', ha='center', va='bottom', color=col)

plt.tight_layout()


# ## 5) Identity Prediction from Scripted vs Spontaneous speech

# Here, we want to see the ability of speech embeddings generated from scripted/spontaneous samples to predict speaker identity and compare both performances.

# In[272]:


#split train and test samples for each participant
spon_df = df.loc[df.Label=='spon']
script_df = df.loc[df.Label=='script']
spon_train=[]; spon_test = []
script_train=[]; script_test = []
for speaker in df['Speaker_ID'].unique():
    speaker_spon_df = spon_df.loc[spon_df.Speaker_ID == speaker]
    speaker_script_df = script_df.loc[script_df.Speaker_ID == speaker]
    msk = np.random.rand(len(speaker_spon_df)) < 0.7
    spon_train.append(speaker_spon_df[msk])
    spon_test.append(speaker_spon_df[~msk])
    script_train.append(speaker_script_df[msk])
    script_test.append(speaker_script_df[~msk])
train_spon_df = pd.concat(spon_train)
test_spon_df = pd.concat(spon_test)
train_script_df = pd.concat(script_train)
test_script_df = pd.concat(script_test)


# In[273]:


train_spon_features = train_spon_df.iloc[:, 4:]
train_spon_labels = train_spon_df['Speaker_ID']
test_spon_features = test_spon_df.iloc[:, 4:]
test_spon_labels = test_spon_df['Speaker_ID']
train_script_features = train_script_df.iloc[:, 4:]
train_script_labels = train_script_df['Speaker_ID']
test_script_features = test_script_df.iloc[:, 4:]
test_script_labels = test_script_df['Speaker_ID']


# ### 5.1 Identity prediction from spontaneous samples

# In[269]:


clf_names, clfs, params_clf = get_sklearn_models()
grid_results = {}
for i, (clf_name, clf, clf_params) in enumerate(zip(clf_names, clfs, params_clf)):
    print(f'Step {i+1}/{len(clf_names)}: {clf_name}...')    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=_RANDOM_SEED)
    pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
    grid_search = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=cv, scoring='recall_macro', error_score=0)
    grid_result = grid_search.fit(train_spon_features, train_spon_labels)
    grid_results[clf_name] = grid_result
    test_result = grid_result.score(test_spon_features, test_spon_labels)
    print(f'Best {clf_name} UAR: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')
    print(f'  Test Data UAR: {test_result*100: .2f}')


# ### 5.2 Identity prediction from scripted samples

# In[274]:


clf_names, clfs, params_clf = get_sklearn_models()
grid_results = {}
for i, (clf_name, clf, clf_params) in enumerate(zip(clf_names, clfs, params_clf)):
    print(f'Step {i+1}/{len(clf_names)}: {clf_name}...')    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=_RANDOM_SEED)
    pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
    grid_search = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=cv, scoring='recall_macro', error_score=0)
    grid_result = grid_search.fit(train_script_features, train_script_labels)
    grid_results[clf_name] = grid_result
    test_result = grid_result.score(test_script_features, test_script_labels)
    print(f'Best {clf_name} UAR: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')
    print(f'  Test Data UAR: {test_result*100: .2f}')


# ## 6) Gender Features in BYOL-S

# ### It is evident how the model is capable of separating gender properly as shown in the [dimensionality reduction plots](#another_cell). Accordingly, we will explore the main BYOL-S features that identify gender and remove them to see if BYOL-S representation would still be capable of maintaining gender separation or would it shed more light on a different kind of acoustic variation.
# 
# ### Methodology:
# 1) Train 3 classifiers (Logistic Regression 'LR', Random Forest 'RF' and Support Vector Classifier 'SVC') to predict gender from BYOL-S embeddings.
# 2) Select the top important features in gender prediction for each trained model.
# 3) Extract the common features across the 3 classifiers.
# 4) Remove these features from the extracted embeddings and apply dimensionality reduction to observe changes.
# 
# ### Model Training: The training process constitutes running 5-fold CV on standardized inputs and reporting the best Recall score.

# ### 6.1 Train Classifiers

# In[15]:


#binarize the gender label
gender_binary = pd.get_dummies(gender)
gender_binary = gender_binary.values
gender_binary = gender_binary.argmax(1)

#define classifiers' objects and fit dataset
clf_names, clfs, params_clf = get_sklearn_models()
grid_results = {}
for i, (clf_name, clf, clf_params) in enumerate(zip(clf_names, clfs, params_clf)):
    print(f'Step {i+1}/{len(clf_names)}: {clf_name}...')    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=_RANDOM_SEED)
    pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
    grid_search = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=cv, scoring='recall_macro', error_score=0)
    grid_result = grid_search.fit(byols_embeddings, gender_binary)
    grid_results[clf_name] = grid_result
    print(f'Best {clf_name} UAR: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')


# ### 6.2 Select the important features for gender prediction

# In[66]:


#select top k features from all classifiers
features = []; k=500
for clf_name in clf_names:
    features_df = eval_features_importance(clf_name, grid_results[clf_name])
    features.append(features_df.index[:k])
#get common features among selected top features
indices = reduce(np.intersect1d, (features[0], features[1], features[2]))
#create one array containing only the common top features (gender features) and another one containing the rest (genderless features)
gender_embeddings = byols_embeddings[:, indices]
genderless_embeddings = np.delete(byols_embeddings, indices, axis=1)


# ### Gender-related features Analysis

# In[15]:


fig, ax = plt.subplots(1, 4, figsize=(40, 15))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=20)
plt.tight_layout()


# In[37]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1600,
    height=600, showlegend=False,)
fig.show()


# In[23]:


fig, ax = plt.subplots(1, 4, figsize=(40, 15))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'id', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()


# In[24]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    max_idx = metrics[optimize].loc[(metrics.Method==name)].idxmax()
    metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
    visualize_embeddings(df, 'id', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1600,
    height=600, showlegend=False,)
fig.show()


# ### *"Genderless"*-related Features Analysis

# In[25]:


fig, ax = plt.subplots(1, 4, figsize=(40, 15))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=20)
plt.tight_layout()


# In[26]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1600,
    height=600, showlegend=False,)
fig.show()


# In[27]:


fig, ax = plt.subplots(1, 4, figsize=(40, 15))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'id', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=20)
plt.tight_layout()


# In[28]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    max_idx = metrics[optimize].loc[(metrics.Method==name)].idxmax()
    metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
    visualize_embeddings(df, 'id', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1600,
    height=600, showlegend=False,)
fig.show()


# ## 7) Acoustic Features Analysis in BYOL-S

# In this section, we will compute some acoustic features (F0 and loudness) from the audio files and see their distribution in the 2D dimensionality reduction plots.

# In[195]:


import pyloudnorm as pyln
f0s = []; loudness = []; mffcc_1 = []; rms=[]
for file in tqdm(wav_files):
    audio, orig_sr = sf.read(file)
    
#     #measure the median fundamental frequency
#     f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C1'),
#                             fmax=librosa.note_to_hz('C7'), sr=orig_sr)
#     f0s.append(np.nanmedian(f0))
    
#     #measure the loudness 
#     meter = pyln.Meter(orig_sr) # create BS.1770 meter
#     l = meter.integrated_loudness(audio)
#     loudness.append(l)
    
#     #measure the first mfcc
#     mfccs = librosa.feature.mfcc(audio, sr=orig_sr)
#     mffcc_1.append(np.nanmedian(mfccs[0,:]))
    
    #measure rms
    rms.append(np.nanmedian(librosa.feature.rms(audio)))


# In[196]:


with open("rms.pickle", "wb") as output_file:
    pickle.dump(rms, output_file)


# In[197]:


with open("f0s.pickle", "rb") as output_file:
    f0s = np.array(pickle.load(output_file))
with open("loudness.pickle", "rb") as output_file:
    loudness = np.array(pickle.load(output_file))
with open("mfcc_1.pickle", "rb") as output_file:
    mfcc_1 = np.array(pickle.load(output_file))
with open("rms.pickle", "rb") as output_file:
    rms = np.array(pickle.load(output_file))


# Plotting the Median F0 of audio samples across 4 dimensionality reduction methods

# In[179]:


fig, ax = plt.subplots(2, 4, figsize=(30, 15))
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    indices = list(np.where(labels == label)[0])
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        points = visualize_embeddings(df, 'f0', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='colorbar')
    ax[i, 0].set_ylabel(label, fontsize=15)
cbar = fig.colorbar(points, ax=ax.ravel().tolist())
cbar.ax.set_ylabel('Median F0', rotation=270)
plt.show()


# In[178]:


fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    indices = list(np.where(labels == label)[0])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'f0', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1600,
    height=1200, showlegend=False,)
fig.show()


# Plotting the Loudness of audio samples across 4 dimensionality reduction methods

# In[180]:


fig, ax = plt.subplots(2, 4, figsize=(30, 15))
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    indices = list(np.where(labels == label)[0])
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        points = visualize_embeddings(df, 'loudness', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='colorbar')
    ax[i, 0].set_ylabel(label, fontsize=15)
cbar = fig.colorbar(points, ax=ax.ravel().tolist())
cbar.ax.set_ylabel('Loudness', rotation=270)
plt.show()


# In[181]:


fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    indices = list(np.where(labels == label)[0])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'loudness', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1600,
    height=1200, showlegend=False,)
fig.show()


# Plotting the median of first MFCC of audio samples across 4 dimensionality reduction methods

# In[191]:


fig, ax = plt.subplots(2, 4, figsize=(30, 15))
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    indices = list(np.where(labels == label)[0])
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]; df['mfcc_1'] = mfcc_1[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        points = visualize_embeddings(df, 'mfcc_1', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='colorbar')
    ax[i, 0].set_ylabel(label, fontsize=15)
cbar = fig.colorbar(points, ax=ax.ravel().tolist())
cbar.ax.set_ylabel('Median MFCC 1', rotation=270)
plt.show()


# In[205]:


fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    indices = list(np.where(labels == label)[0])
    df['f0'] = f0s[indices]; df['loudness'] = loudness[indices]; df['mfcc_1'] = mfcc_1[indices]; df['rms'] = rms[indices]
    df['f0'] = df['f0'].mask(df['f0'] > 300, 300)
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'mfcc_1', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1600,
    height=1200, showlegend=False,)
fig.show()


# Plotting the median of RMS of audio samples across 4 dimensionality reduction methods

# In[199]:


fig, ax = plt.subplots(2, 4, figsize=(30, 15))
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    indices = list(np.where(labels == label)[0])
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df['rms'] = rms[indices]
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        points = visualize_embeddings(df, 'rms', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='colorbar')
    ax[i, 0].set_ylabel(label, fontsize=15)
cbar = fig.colorbar(points, ax=ax.ravel().tolist())
cbar.ax.set_ylabel('Median RMS', rotation=270)
plt.show()


# In[200]:


fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
unique_labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(unique_labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    indices = list(np.where(labels == label)[0])
    df['rms'] = rms[indices]
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'rms', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1600,
    height=1200, showlegend=False,)
fig.show()

