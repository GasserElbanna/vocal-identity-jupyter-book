#!/usr/bin/env python
# coding: utf-8

# # Analysis for compiling behavioral experiment stimuli

# The aim of this notebook is to prepare the stimuli that will be used in the behavioral experiment. To do so, we perform some analyses on datasets to evaluate how a model can *place* a speaker in its perceptual space. How do models encode speakers' identities? How far are these encoded representations from each other? Would the distances between speaker in this space correlate to the ability of a model to recognize them properly? e.g. the further the speakers from each other the easier for the model to discriminate.

# Accordingly, we compute different metrics across speakers' utterances to first show correlations between recognizability and distance/similarity variables. Then, indicate which variable is more convenient to compile the stimuli based on it.

# In[4]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
import deciphering_enigma
import matplotlib.pyplot as plt


# In[2]:


import matplotlib
from pylab import cm
import matplotlib as mpl
matplotlib.font_manager._fmcache
matplotlib.font_manager._rebuild()
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 3


# ## TIMIT

# ### 1) Load Dataset with Metadata

# In[3]:


#define the experiment config file path
path_to_config = './config.yaml'

#read the experiment config file
exp_config = deciphering_enigma.load_yaml_config(path_to_config)
dataset_path = exp_config.dataset_path

#register experiment directory and read wav files' paths
audio_files = deciphering_enigma.build_experiment(exp_config)
audio_files = [audio for audio in audio_files if audio.endswith('_normloud.wav')]
print(f'Dataset has {len(audio_files)} samples')

#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
metadata_df['AudioPath'] = audio_files
metadata_df['ID'] = np.array(list(map(lambda x: x.split('/')[-2][1:], audio_files)))
metadata_df['Gender'] = np.array(list(map(lambda x: x.split('/')[-2][0], audio_files)))

dur = []
for file in tqdm(audio_files):
    audio, sr = sf.read(file)
    assert sr == 16000
    dur.append(len(audio)/sr)
metadata_df['Duration'] = dur
metadata_df


# ### 2) Preprocessing Data

# ### (Select Duration from 2.8 to 3s/Normalize Loudness/Convert to tensors)

# In[6]:


metadata_df = metadata_df.loc[(metadata_df.Duration <= 3) & (metadata_df.Duration >= 2.8)]
print(f'Number of speakers: {metadata_df.ID.unique().shape[0]}')
print(f'Number of utterances: {len(metadata_df)}')


# In[14]:


#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(list(metadata_df.AudioPath), cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format, 
                                                    norm_loudness=exp_config.norm_loudness, target_loudness=exp_config.target_loudness)


# ### 3) Extract Speech Embeddings

# In[16]:


#generate speech embeddings
feature_extractor = deciphering_enigma.FeatureExtractor()
embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)


# ### 4) Cosine Distance

# We computed the cosine distances across the embeddings extracted from all utterances in TIMIT (6300) using the best-performing layer in **HuBERT**. Then, we averaged the distances for each speaker to evaluate how *far* one speaker from the rest. The results are shown in the form of a heatmap for all 630 speakers and their distances between each other.

# In[5]:


#Load the disimlarity measures across speakers in a long form
df_long = pd.read_csv(f'../{exp_config.dataset_name}/HuBERT_best/longform_cosine.csv')
#Average dissimilarity values for each speaker-speaker pair
df_averaged = pd.DataFrame(df_long.groupby(['ID_1', 'ID_2'])['Distance'].mean()).reset_index()
#Convert long form df to pairwise
df_cosine = df_averaged.pivot(index='ID_1', columns='ID_2', values='Distance')


# In[6]:


#Plot RDM across all speakers
fig, ax = plt.subplots(1, 1, figsize=(25, 20))
ax = sns.heatmap(df_cosine, ax=ax)
ax.figure.axes[-1].set_ylabel('cosine distance', size=40)
ax.set_xlabel('Speakers', fontsize=40)
ax.set_ylabel('Speakers', fontsize=40)
plt.tight_layout()


# The speakers in the y-axis are then sorted from the highest distance to lowest, as shown below. Thus, the speaker on the top is the furthest on average to all speakers, meaning that this speaker is more distinct or unique. In simpler terms, the speakers are sorted from unique voices to more common and average voices.

# In[7]:


#Plot more-least common voice matrix
df_cosine_sorted = df_cosine.assign(m=df_cosine.mean(axis=1)).sort_values('m', ascending=False).drop('m', axis=1)
fig, ax = plt.subplots(1, 1, figsize=(25, 20))
ax = sns.heatmap(df_cosine_sorted, ax=ax)
ax.figure.axes[-1].set_ylabel('cosine distance', size=40)
ax.set_xlabel('Speakers', fontsize=40)
ax.set_ylabel('Speakers', fontsize=40)
plt.tight_layout()


# Below we plot the distribution of distances that were computed between different speakers against the ones computed between same speaker utterances.

# In[8]:


#Add column for same and different speakers
df_averaged['Speaker'] = df_averaged.apply(lambda x: 'Same' if x.ID_1 == x.ID_2 else 'Different', axis=1)
df_unique = df_averaged.drop_duplicates('Distance')

df_same = df_unique.loc[df_unique['Speaker'] == 'Same']
#Sample a rondom set from the distances from different speakers because the number of different is higher than same samples
df_diff =df_unique.loc[df_unique['Speaker'] == 'Different'].sample(len(df_unique.loc[df_unique['Speaker'] == 'Same']), random_state=42)
df_concat = pd.concat([df_same, df_diff])

#Plot histogram for model
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
hist = sns.histplot(data=df_concat, x='Distance', hue='Speaker', ax=ax, kde=True, stat='density')
ax.set_ylabel('Density')
ax.set_xlabel('Cosine Distance')
plt.tight_layout()


# ### 5) Representation Dissimilarity Matrix (RDM)

# Now, we compute a dissimilarity matrix which basically calculates $1 - spearman$ correlation. It is another metric to evaluate how close/far are speaker's representations from others.

# In[9]:


#Load the disimlarity measures across speakers in a long form
df_long = pd.read_csv(f'../{exp_config.dataset_name}/HuBERT_best/longform_inverse_spearman.csv')
#Average dissimilarity values for each speaker-speaker pair
df_averaged = pd.DataFrame(df_long.groupby(['ID_1', 'ID_2'])['Distance'].mean()).reset_index()
#Convert long form df to pairwise
df_rdm = df_averaged.pivot(index='ID_1', columns='ID_2', values='Distance')


# In[10]:


#Plot RDM across all speakers
fig, ax = plt.subplots(1, 1, figsize=(25, 20))
ax = sns.heatmap(df_rdm, ax=ax)
ax.figure.axes[-1].set_ylabel('1-spearman', size=40)
ax.set_xlabel('Speakers', fontsize=40)
ax.set_ylabel('Speakers', fontsize=40)
plt.tight_layout()


# In a similar vein, we sort the speakers from higher $1-spearman$ (lower similarity) values to lower ones (higher similarity) on average. That way we could also present speakers from unique to common as well.

# In[11]:


#Plot least-more common voice matrix
df_rdm_sorted = df_rdm.assign(m=df_rdm.mean(axis=1)).sort_values('m', ascending=False).drop('m', axis=1)
fig, ax = plt.subplots(1, 1, figsize=(25, 20))
ax = sns.heatmap(df_rdm_sorted, ax=ax)
ax.figure.axes[-1].set_ylabel('1-spearman', size=40)
ax.set_xlabel('Speakers', fontsize=40)
ax.set_ylabel('Speakers', fontsize=40)
plt.tight_layout()


# Also, we plot the distributions of $1-spearman$ in case of same and different speakers

# In[12]:


#Add column for same and different speakers
df_averaged['Speaker'] = df_averaged.apply(lambda x: 'Same' if x.ID_1 == x.ID_2 else 'Different', axis=1)
df_unique = df_averaged.drop_duplicates('Distance')

df_same = df_unique.loc[df_unique['Speaker'] == 'Same']
df_diff =df_unique.loc[df_unique['Speaker'] == 'Different'].sample(len(df_unique.loc[df_unique['Speaker'] == 'Same']), random_state=42)
df_concat = pd.concat([df_same, df_diff])

#Plot histogram for model
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
hist = sns.histplot(data=df_concat, x='Distance', hue='Speaker', ax=ax, kde=True, stat='density')
ax.set_ylabel('Density')
ax.set_xlabel('1-Spearman')
plt.tight_layout()


# ### 6) Confusion Matrix

# Another way to evaluate how the model perceive speakers' identities and how close/far the model encode voices in its perceptual space is by evaluating which speakers the model gets confused between. This might be considered as a proxy for distances between speakers, the closer they are the more difficult for the model to recognize them. Accordingly, we computed a confusion matrix to show how many times the model misidentified one speaker with another. We implemented this experiment by bootstrapping the voice recognition task on HuBERT best layer representations for 10,000 trials. Each trial we randomly sample 7 utterances from each speaker for training and keep 3 for testing. We saved the predictions for each trial in addition to the ground truth. Then count the times the model misidentified each speaker with another speaker and plot it as heatmap.

# In[13]:


from ast import literal_eval
df_bootstrap = pd.read_csv(f'../{exp_config.dataset_name}/HuBERT_best/HuBERT_best_bootstrap_predictions.csv')
df_bootstrap['ID'] = df_bootstrap['ID'].apply(lambda x: literal_eval(x))
df_bootstrap['Pred'] = df_bootstrap['Pred'].apply(lambda x: literal_eval(x))


# In[14]:


df_bootstrap_long = df_bootstrap.explode(['ID', 'Pred']).drop(columns=['Unnamed: 0'])
#Remove the true Identifications (model prediction == ground truth)
df_bootstrap_long = df_bootstrap_long.loc[df_bootstrap_long.ID != df_bootstrap_long.Pred]
df_bootstrap_long['count'] = 1
df_bootstrap_long = df_bootstrap_long.groupby(['ID', 'Pred']).count().reset_index()
#Compute the log of counts for better visualization
df_bootstrap_long['log_count'] = np.log(df_bootstrap_long['count'])
df_bootstrap_pairwise = df_bootstrap_long.pivot(index='ID', columns='Pred', values='log_count').fillna(0)
df_bootstrap_sorted_pairwise = df_bootstrap_pairwise.assign(m=df_bootstrap_pairwise.sum(axis=1)).sort_values('m').drop('m', axis=1)


# Below we plot the confusion matrix with speakers sorted from least to most false identifications meaning that top speakers on the y-axis were relatively easy for the model to identify, hence, unique. While the speakers in the bottom were more difficult to the model to properly identify.

# In[15]:


#Plot confusion matrix across all speakers
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax = sns.heatmap(df_bootstrap_sorted_pairwise, ax=ax)
ax.figure.axes[-1].set_ylabel('log(Count)', size=30)
ax.set_xlabel('Speakers', fontsize=30)
ax.set_ylabel('Speakers', fontsize=30)
plt.tight_layout()


# In[59]:


#Sample from the speaker with least amount of misidentifications
import IPython
IPython.display.Audio(list(df.loc[df.ID == 'AKS0']['AudioPath'])[0])


# In[66]:


#Sample from the speaker (WAD0) with most amount of misidentifications
import IPython
IPython.display.Audio(list(df.loc[df.ID == 'WAD0']['AudioPath'])[0])


# In[65]:


#Sample from the speaker that was mostly confused with WAD0
speaker = df_bootstrap_sorted_pairwise['WAD0'].idxmax()
IPython.display.Audio(list(df.loc[df.ID == speaker]['AudioPath'])[0])


# ### 7) Relations between measured variables

# It is worth mentioning that the TIMIT dataset comprises utterances with variable durations. The duration average of clips is $3.1±0.8$ seconds. Accordingly, it is important to examine if clip duration is a confounding variable for model's performance. Thus, in this subsection, we plotted the relation of the variables measured above with duration and each other.

# In[16]:


#Get the average cosine values for each speaker
cosine = list(df_cosine_sorted.mean(axis=1))
ID = list(df_cosine_sorted.index)
df_cos = pd.DataFrame({'ID': ID, 'Cosine': cosine})

#Get the average RDM values for each speaker
rdm = list(df_rdm_sorted.mean(axis=1))
ID = list(df_rdm_sorted.index)
df_spear = pd.DataFrame({'ID': ID, '1-Spearman': rdm})

#Get the average misidentification values for each speaker
conf = list(df_bootstrap_long.groupby('ID')['count'].sum())
ID = list(df_bootstrap_sorted_pairwise.index)
df_conf = pd.DataFrame({'ID': ID, 'MisID_count': conf})

#Merge all variables in one dataframe on ID
df = df_cos.merge(df_spear, on='ID')
df = df.merge(df_conf, on='ID')
df = df.merge(metadata_df, on='ID')


# In[17]:


duration = list(df.groupby('ID')['Duration'].mean())
df = df.drop_duplicates('Cosine')


# #### **Duration**

# In[43]:


import scipy.stats as stats
#Plot the relation between duration and all variables
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
ax = ax.flatten()
sns.regplot(data=df, y='Duration', x='MisID_count', ax=ax[0])
r, p = stats.pearsonr(df['Duration'], df['MisID_count'])
ax[0].text(df['MisID_count'].mean(), df['Duration'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[0].set_xlabel('Misidentification', fontsize=20)
ax[0].set_ylabel('Duration', fontsize=20)
plot = sns.regplot(data=df, y='Duration', x='Cosine', ax=ax[1])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['Duration'], df['Cosine'])
ax[1].text(df['Cosine'].mean(), df['Duration'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[1].set_xlabel('Cosine Distance', fontsize=20)
plot = sns.regplot(data=df, y='Duration', x='1-Spearman', ax=ax[2])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['Duration'], df['1-Spearman'])
ax[2].text(df['1-Spearman'].mean(), df['Duration'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[2].set_xlabel('1-Spearman', fontsize=20)
plt.tight_layout()


# #### **Misidentification**

# In[45]:


import scipy.stats as stats
#Plot the relation between duration and all variables
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
ax = ax.flatten()
sns.regplot(data=df, y='MisID_count', x='Duration', ax=ax[0])
r, p = stats.pearsonr(df['MisID_count'], df['Duration'])
ax[0].text(df['Duration'].mean(), df['MisID_count'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[0].set_xlabel('Duration', fontsize=20)
ax[0].set_ylabel('Misidentification', fontsize=20)
plot = sns.regplot(data=df, y='MisID_count', x='Cosine', ax=ax[1])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['MisID_count'], df['Cosine'])
ax[1].text(df['Cosine'].mean(), df['MisID_count'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[1].set_xlabel('Cosine Distance', fontsize=20)
plot = sns.regplot(data=df, y='MisID_count', x='1-Spearman', ax=ax[2])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['MisID_count'], df['1-Spearman'])
ax[2].text(df['1-Spearman'].mean(), df['MisID_count'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[2].set_xlabel('1-Spearman', fontsize=20)
plt.tight_layout()


# #### **Cosine Distances**

# In[49]:


import scipy.stats as stats
#Plot the relation between duration and all variables
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
ax = ax.flatten()
sns.regplot(data=df, y='Cosine', x='Duration', ax=ax[0])
r, p = stats.pearsonr(df['Cosine'], df['Duration'])
ax[0].text(df['Duration'].mean(), df['Cosine'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[0].set_xlabel('Duration', fontsize=20)
ax[0].set_ylabel('Cosine Distance', fontsize=20)
plot = sns.regplot(data=df, y='Cosine', x='MisID_count', ax=ax[1])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['MisID_count'], df['Cosine'])
ax[1].text(df['MisID_count'].mean(), df['Cosine'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[1].set_xlabel('Misidentification', fontsize=20)
plot = sns.regplot(data=df, y='Cosine', x='1-Spearman', ax=ax[2])
plot.set(ylabel=None)
r, p = stats.pearsonr(df['Cosine'], df['1-Spearman'])
ax[2].text(df['1-Spearman'].mean(), df['Cosine'].max(), f"r={r:.2f}, p={p:.2f}", fontsize=15)
ax[2].set_xlabel('1-Spearman', fontsize=20)
plt.tight_layout()


# It looks like the duration of utterances showed very low correlation with the number of misidentifications and cosine distances. However, there was a slight correlation between duration and 1-spearman variable. That might indicate that the spearman metric is affected by the duration of the clip more than the uniqueness and closeness of voices. Especially, that it showed very low correlation with misidentifications. Also, 1-spearman showed high correlation with cosine distances meaning that the further the representations the lower the similarity which is intuitive but doesn't look like the same behavior is shown between duration and cosine distances. That said, we could rely on cosine distances and number of misidentifications as proxy for how speakers information is encoded in a model's perceptual space.
