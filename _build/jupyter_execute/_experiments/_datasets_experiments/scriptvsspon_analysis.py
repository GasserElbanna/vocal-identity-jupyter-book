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

# The dataset was processed by undersampling to 16 kHz to be compatible with BYOL-S model. Additionally, the utterances were cropped to fixed durations (1, 3, 5, 10, 15 sec) to yield 5 new datasets generated from the original one.
# 
# Finally, the naming convention for the audio files is: *{ID}_{Gender}_{Task}_{Label}_{File Number}.wav* (e.g. 049_F_DHR_script_000.wav).

# In the following analysis, we will be using the 3sec-utterance version of the dataset.

# ## 1) Loading Data

# In[1]:


#read wav files' paths
wav_dirs = sorted(glob('datasets/scripted_spont_dataset/preprocessed_audios_dur3sec/*'))
wav_files = sorted(glob('datasets/scripted_spont_dataset/preprocessed_audios_dur3sec/*/*.wav'))
print(f'{len(wav_files)} samples')


# In[10]:


#balancing the number of audio files in each label (i.e. to have equal number of scripted vs spontaneous samples per subject)
files = []
for wav_dir in wav_dirs:
    wav_files = np.array(sorted(glob(f'{wav_dir}/*.wav')))
    ids = np.array(list(map(lambda x: os.path.basename(x).split('_')[0], wav_files)))
    labels = np.array(list(map(lambda x: os.path.basename(x).split('_')[3], wav_files)))
    min_label = min(Counter(labels).values())
    script_files = [file for file in wav_files if os.path.basename(file).split('_')[3] == 'script'][:min_label]
    spon_files = [file for file in wav_files if os.path.basename(file).split('_')[3] == 'spon'][:min_label]
    files += spon_files + script_files
wav_files = files


# In[11]:


#extract metadata from path (Script VS Spon data)
wav_names = np.array(list(map(lambda x: os.path.basename(x), wav_files)))
gender = np.array(list(map(lambda x: os.path.basename(x).split('_')[1], wav_files)))
speaker_ids = np.array(list(map(lambda x: os.path.basename(x).split('_')[0], wav_files)))
labels = np.array(list(map(lambda x: os.path.basename(x).split('_')[3], wav_files)))


# In[12]:


#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = load_dataset(wav_files)
len(audio_tensor_list)


# ## 2) Generating BYOL-S Embeddings

# In order to generate speech embeddings using BYOL-S model, we installed our [package](https://github.com/GasserElbanna/serab-byols) to extract the needed features and use the model checkpoints.

# In[13]:


#generate speech embeddings
_CHECKPOINT_PATH = 'serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth'
_CONFIG_PATH = 'serab-byols/serab_byols/config.yaml'
_MODEL_NAME = 'default'
byols_embeddings = generate_speech_embeddings(audio_tensor_list, model_name=_MODEL_NAME, config_path=_CONFIG_PATH, checkpoint_path=_CHECKPOINT_PATH)
byols_embeddings.shape


# ## 3) Analysis in High dimension

# ### 3.1 Prepare data for computing cosine distances in the original dimensions

# In[234]:


#create dataframe with all dataset metadata
data = {'Speaker_ID':speaker_ids, 'Gender':gender, 'Label':labels, 'Audio_File':wav_names}
df = pd.DataFrame(data=data)
df.head()


# In[235]:


#add embeddings to original dataframe
df_embeddings = pd.DataFrame(byols_embeddings)
df_embeddings = df_embeddings.add_prefix('Embeddings_')
df = pd.concat([df, df_embeddings], axis=1)
df.head()


# In[9]:


#create distance-based dataframe between all data samples in a square form
pairwise = pd.DataFrame(
    squareform(pdist(df.iloc[:, 5:], metric='cosine')),
    columns = df['Audio_File'],
    index = df['Audio_File']
)


# In[10]:


#move from square form DF to long form DF
long_form = pairwise.unstack()
#rename columns and turn into a dataframe
long_form.index.rename(['Sample_1', 'Sample_2'], inplace=True)
long_form = long_form.to_frame('Distance').reset_index()
#remove the distances computed between same samples (distance = 0)
long_form = long_form.loc[long_form['Sample_1'] != long_form['Sample_2']]
long_form.sample(10)


# In[11]:


#add columns for meta-data
long_form['Gender'] = long_form.apply(lambda row: row['Sample_1'].split('_')[1] if row['Sample_1'].split('_')[1] == row['Sample_2'].split('_')[1] else 'Different', axis=1)
long_form['Label'] = long_form.apply(lambda row: row['Sample_1'].split('_')[3] if row['Sample_1'].split('_')[3] == row['Sample_2'].split('_')[3] else 'Different', axis=1)
long_form['ID'] = long_form.apply(lambda row: row['Sample_1'].split('_')[0] if row['Sample_1'].split('_')[0] == row['Sample_2'].split('_')[0] else 'Different', axis=1)
long_form.sample(10)


# In[12]:


#remove distances computed between different speakers and different labels
df = long_form.loc[(long_form['Gender']!='Different') & (long_form['Label']!='Different') & (long_form['ID']!='Different')]
df.sample(10)


# ### 3.2 Per Speaker Analysis

# #### Here, we explore the differences in cosine distances for each speaker based on the labels (in this case scripted vs spontaneous speech).

# In[19]:


speakers = df['ID'].unique()
fig, ax = plt.subplots(5, 6, figsize=(40, 40))
ax = ax.flatten()
for i, speaker in enumerate(speakers):
    speaker_df = df.loc[(df['Sample_1'].str.contains(f'{speaker}_')) & (df['Sample_2'].str.contains(f'{speaker}_'))]
    sns.violinplot(data=speaker_df, x='Label', y='Distance', inner='quartile', ax=ax[i])
    ax[i].set(xlabel=None, ylabel=None)
    ax[i].set_title(f'Speaker {speaker}')
    
    # statistical annotation
    d=cohend(speaker_df['Distance'].loc[(speaker_df.Label=='spon')], speaker_df['Distance'].loc[(speaker_df.Label=='script')])
    x1, x2 = 0, 1
    y, h, col = speaker_df['Distance'].max() + speaker_df['Distance'].max()*0.05, speaker_df['Distance'].max()*0.01, 'k'
    ax[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax[i].text((x1+x2)*.5, y+(h*1.5), f'cohen d={d:.2}', ha='center', va='bottom', color=col)


fig.text(0.5, -0.01, 'Labels', ha='center', fontsize=30)
fig.text(-0.01, 0.5, 'Cosine Distances', va='center', rotation='vertical', fontsize=30)
for empty_ax in ax[speakers.shape[0]:]:
    empty_ax.set_visible(False)
plt.tight_layout()
plt.savefig('perspeaker_scriptvsspon_cosinedist.png')


# ### 3.3 Gender Analysis

# #### Here, we explore the gender effect on embeddings distances.

# In[20]:


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.violinplot(data=df, x='Label', y='Distance', hue='Gender', inner='quartile', split=True, ax=ax)
ax.set_xlabel('Labels', fontsize=15)
ax.set_ylabel('Cosine Distances', fontsize=15)

# statistical annotation
d1=cohend(df['Distance'].loc[(df.Label=='spon')&(df.Gender=='F')], df['Distance'].loc[(df.Label=='spon')&(df.Gender=='M')])
d2=cohend(df['Distance'].loc[(df.Label=='script')&(df.Gender=='F')], df['Distance'].loc[(df.Label=='script')&(df.Gender=='M')])
x1, x2, x3, x4 = -0.25, 0.25, 0.75, 1.25
y, h, col = df['Distance'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.plot([x3, x3, x4, x4], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+(h*1.5), f'cohen d={d1:.2}', ha='center', va='bottom', color=col)
plt.text((x3+x4)*.5, y+(h*1.5), f'cohen d={d2:.2}', ha='center', va='bottom', color=col)

plt.tight_layout()
plt.savefig('gender_scriptvsspon_cosinedist.png')


# ### 3.4 Overall Label Analysis

# In[21]:


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
plt.savefig('overalllabel_scriptvsspon_cosinedist.png')


# ## <span style="color:red"><div style="text-align: justify">Let's try the same *overall label analysis* but with Hybrid BYOL-S model instead of BYOL-S.</div></span>

# In[6]:


#generate speech embeddings
_CHECKPOINT_PATH = 'serab-byols/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth'
_CONFIG_PATH = 'serab-byols/serab_byols/config.yaml'
_MODEL_NAME = 'cvt'
hybrid_byols_embeddings = generate_speech_embeddings(audio_tensor_list, model_name=_MODEL_NAME, config_path=_CONFIG_PATH, checkpoint_path=_CHECKPOINT_PATH)
hybrid_byols_embeddings.shape


# In[7]:


#create dataframe with all dataset metadata
data = {'Speaker_ID':speaker_ids, 'Gender':gender, 'Label':labels, 'Audio_File':wav_names}
df = pd.DataFrame(data=data)

#add embeddings to original dataframe
df_embeddings = pd.DataFrame(hybrid_byols_embeddings)
df_embeddings = df_embeddings.add_prefix('Embeddings_')
df = pd.concat([df, df_embeddings], axis=1)
df.head()


# In[8]:


#create distance-based dataframe between all data samples in a square form
pairwise = pd.DataFrame(
    squareform(pdist(df.iloc[:, 5:], metric='cosine')),
    columns = df['Audio_File'],
    index = df['Audio_File']
)
#move from square form DF to long form DF
long_form = pairwise.unstack()
#rename columns and turn into a dataframe
long_form.index.rename(['Sample_1', 'Sample_2'], inplace=True)
long_form = long_form.to_frame('Distance').reset_index()
#remove the distances computed between same samples (distance = 0)
long_form = long_form.loc[long_form['Sample_1'] != long_form['Sample_2']]
#add columns for meta-data
long_form['Gender'] = long_form.apply(lambda row: row['Sample_1'].split('_')[1] if row['Sample_1'].split('_')[1] == row['Sample_2'].split('_')[1] else 'Different', axis=1)
long_form['Label'] = long_form.apply(lambda row: row['Sample_1'].split('_')[3] if row['Sample_1'].split('_')[3] == row['Sample_2'].split('_')[3] else 'Different', axis=1)
long_form['ID'] = long_form.apply(lambda row: row['Sample_1'].split('_')[0] if row['Sample_1'].split('_')[0] == row['Sample_2'].split('_')[0] else 'Different', axis=1)
#remove distances computed between different speakers and different labels
df = long_form.loc[(long_form['Gender']!='Different') & (long_form['Label']!='Different') & (long_form['ID']!='Different')]
df.sample(10)


# In[10]:


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


# ## 4) Dimensionality Reduction

# #### The previous analysis showed how well the model is capable of grouping the uttereances of the same speaker in different cases (scripted and spontaneous) in the embedding space (high dimension). That being said, we will replicate the same analysis but in the lower dimension space to visualize the impact of speaking styles on voice identity perception.

# #### Accordingly, we will utilize different kind of dimensionality reduction such as PCA, tSNE, UMAP and PaCMAP to get a better idea of how the speakers' samples are clustered together in 2D. However, one constraint is that these methods are sensitive to their hyperparameters (except PCA) which could imapct our interpretation of the results. Thus, a grid search across the hyperparameters for each method is implemented.

# #### Another issue would be quantifying the ability of these methods to perserve the distances amongst samples in the high dimension and present it in a lower dimension. To address this, we are using two metrics KNN and CPD that represent the ability of the algorithm to preserve local and global structures of the original embedding space, respectively. Both metrics are adopted from this [paper](https://www.nature.com/articles/s41467-019-13056-x) in which they define both metrics as follows:
# 
# * KNN: The fraction of k-nearest neighbours in the original high-dimensional data that are preserved as k-nearest neighbours in the embedding. KNN quantifies preservation of the local, or microscopic structure. The value of K used here is the min number of samples a speaker would have in the original space.
# 
# * CPD: Spearman correlation between pairwise distances in the high-dimensional space and in the embedding. CPD quantifies preservation of the global, or macroscropic structure. Computed across all pairs among 1000 randomly chosen points with replacement. 

# #### Consequently, we present the results from dimensionality reduction methods in two ways, one optimimizing local structure metric (KNN) and the other optimizing global structure metric (CPD).

# ### <a id='another_cell'></a> 4.1 Dimensionality Reduction Methods Comparison

# In[79]:


script_df = pd.read_csv('script_dataset.csv', header=[0,1,2])
script_df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
spon_df = pd.read_csv('spon_dataset.csv', header=[0,1,2])
spon_df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
metrics = pd.read_csv('scriptvsspon_metrics.csv')


# In[11]:


fig, ax = plt.subplots(2, 4, figsize=(40, 20))
optimize = 'Global'
labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'gender', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='sns')
    ax[i, 0].set_ylabel(label, fontsize=15)
ax[i,j].legend(bbox_to_anchor=(1.2, 1.15), fontsize=12)
plt.tight_layout()


# In[29]:


import plotly
from IPython.display import display
from IPython.display import IFrame
# from html import HTML
fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'gender', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1800,
    height=1200, showlegend=False,)
# fig.show()
plotly.offline.plot(fig, filename = 'figure_1.html')
IFrame('figure_1.html', width=1800, height=1200)


# In[13]:


fig, ax = plt.subplots(2, 4, figsize=(40, 20))
optimize = 'Global'
labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'id', metrics=metric, axis=ax[i, j], opt_structure=optimize, red_name=name, plot_type='sns')
    ax[i, 0].set_ylabel(label, fontsize=15)
ax[i,j].legend(bbox_to_anchor=(1.2, 1.5), fontsize=12)
plt.tight_layout()


# In[19]:


fig = make_subplots(rows=2, cols=4)
optimize = 'Global'
labels = ['script', 'spon']
metrics = pd.read_csv('scriptvsspon_metrics.csv')
reducer_names, params_list = get_reducers_params()
for i, label in enumerate(labels):
    df = pd.read_csv(f'{label}_dataset.csv', header=[0,1,2])
    df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
    for j, name in enumerate(reducer_names):
        max_idx = metrics[optimize].loc[(metrics.Protocol==label)&(metrics.Method==name)].idxmax()
        metric = [metrics['Local'].iloc[max_idx], metrics['Global'].iloc[max_idx]]
        visualize_embeddings(df, 'id', metrics=metric, axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=i+1, col=j+1, hovertext=df['wav_file'], label=label)
fig.update_layout(
    autosize=False,
    width=1800,
    height=1200, showlegend=False,)
fig.show()


# In[182]:


df = pd.read_csv('scriptvsspon_metrics.csv')
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax = ax.T.flatten()
sns.violinplot(data=df, x='Method', y='Local', hue='Protocol', ax=ax[0], split=True, inner='quartile')
ax[0].set_ylabel('Local Structure', fontsize=15)
ax[0].set_xlabel('Method', fontsize=15)
ax[0].set_ylim([0,1])
add_stat_annotation(ax[0], data=df, x='Method', y='Local', hue='Protocol',
                    box_pairs=[
                                 (("tSNE", 'script'), ("tSNE", 'spon')),
                                 (("UMAP", 'script'), ("UMAP", 'spon')),
                                 (("PaCMAP", 'script'), ("PaCMAP", 'spon'))
                                ],
                    test='t-test_ind', text_format='star', loc='inside', verbose=0)

sns.violinplot(data=df, x='Method', y='Global', hue='Protocol', ax=ax[1], split=True, inner='quartile')
ax[1].set_ylabel('Global Structure', fontsize=15)
ax[1].set_xlabel('Method', fontsize=15)
ax[1].set_ylim([0,1])
add_stat_annotation(ax[1], data=df, x='Method', y='Global', hue='Protocol',
                    box_pairs=[
                                 (("tSNE", 'script'), ("tSNE", 'spon')),
                                 (("UMAP", 'script'), ("UMAP", 'spon')),
                                 (("PaCMAP", 'script'), ("PaCMAP", 'spon'))
                                ],
                    test='t-test_ind', text_format='star', loc='inside', verbose=0)
fig.tight_layout()


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

# In[29]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', plot_type='sns', red_name='PCA', row=1, col=1, hovertext='', label='spon'):
    if plot_type == 'sns':
        sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name
                        , style=label_name, palette='deep', ax=axis)
        axis.set(xlabel=None, ylabel=None)
        axis.get_legend().remove()
        if len(metrics) != 0:
            axis.set_title(f'{red_name}: KNN={metrics[0]:0.2f}, CPD={metrics[1]:0.2f}', fontsize=15)
        else:
            axis.set_title(f'{red_name}', fontsize=15)
    else:
        traces = px.scatter(x=df[red_name, opt_structure, 'Dim1'], y=df[red_name, opt_structure, 'Dim2'], color=df[label_name].astype(str), hover_name=hovertext, title=f'{red_name}: KNN={metrics[0]:0.2f}, CPD={metrics[1]:0.2f}')
        traces.layout.update(showlegend=False)
        axis.add_traces(
            list(traces.select_traces()),
            rows=row, cols=col
        )


# In[31]:


fig, ax = plt.subplots(1, 4, figsize=(40, 10))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()


# In[61]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1800,
    height=600, showlegend=False,)
fig.show()


# In[32]:


fig, ax = plt.subplots(1, 4, figsize=(40, 10))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'gender_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'id', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()


# In[63]:


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
    width=1800,
    height=600, showlegend=False,)
fig.show()


# ### *"Genderless"*-related Features Analysis

# In[36]:


def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', plot_type='sns', red_name='PCA', row=1, col=1, hovertext='', label='spon'):
    if plot_type == 'sns':
        sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name
                        , style=label_name, palette='deep', ax=axis)
        axis.set(xlabel=None, ylabel=None)
        axis.get_legend().remove()
        if len(metrics) != 0:
            axis.set_title(f'{red_name}: KNN={metrics[0]:0.2f}, CPD={metrics[1]:0.2f}', fontsize=15)
        else:
            axis.set_title(f'{red_name}', fontsize=15)
    else:
        traces = px.scatter(x=df[red_name, opt_structure, 'Dim1'], y=df[red_name, opt_structure, 'Dim2'], color=df[label_name].astype(str), hover_name=hovertext, title=f'{red_name}: KNN={metrics[0]:0.2f}, CPD={metrics[1]:0.2f}')
        traces.layout.update(showlegend=False)
        axis.add_traces(
            list(traces.select_traces()),
            rows=row, cols=col
        )


# In[37]:


fig, ax = plt.subplots(1, 4, figsize=(40, 10))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()


# In[62]:


fig = make_subplots(rows=1, cols=4)
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv('genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'gender', axis=fig, opt_structure=optimize, red_name=name, plot_type='plotly', row=1, col=j+1, hovertext=df['wav_file'])
fig.update_layout(
    autosize=False,
    width=1800,
    height=600, showlegend=False,)
fig.show()


# In[38]:


fig, ax = plt.subplots(1, 4, figsize=(40, 10))
optimize = 'Global'
reducer_names, params_list = get_reducers_params()
df = pd.read_csv(f'genderless_features_scriptvsspon_dataset_217.csv', header=[0,1,2])
df.rename(columns={'Unnamed: 17_level_1': '', 'Unnamed: 17_level_2': '', 'Unnamed: 18_level_1': '', 'Unnamed: 18_level_2': '', 'Unnamed: 19_level_1': '', 'Unnamed: 19_level_2': ''},inplace=True)
for j, name in enumerate(reducer_names):
    visualize_embeddings(df, 'id', axis=ax[j], opt_structure=optimize, red_name=name, plot_type='sns')
ax[j].legend(bbox_to_anchor=(1, 1), fontsize=12)
plt.tight_layout()


# In[64]:


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
    width=1800,
    height=600, showlegend=False,)
fig.show()

