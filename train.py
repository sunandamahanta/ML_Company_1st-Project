#!/usr/bin/env python
# coding: utf-8

# ## Higgs Boson Classification Model

# In[1]:


# Importing Pandas and NumPy
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#data
df = pd.read_csv("Dataset.csv",nrows=30000)
df.head()


# In[3]:


#Number of rows and columns
df.shape


# In[4]:


#Replacing the meaningless values with mean value
df.replace(-999,np.nan,inplace=True)
df.replace(np.nan,df.mean(),inplace=True)


# In[5]:


df.head()


# In[6]:


#Check for data types and null values of columns
df.info()


# ##### There is no null values in the columns.

# In[7]:


#Check for duplicate values
df.duplicated(subset='EventId').sum()


# ##### There is no duplicate value in the id column. We can drop that.

# In[8]:


df.drop('EventId',axis=1,inplace=True)


# In[9]:


#Summary of columns (mean/mode/median)
df.describe()


# ### Exploratory Data Analysis

# In[10]:


import dabl
dabl.plot(df,'Label')


# In[11]:


dabl.plot(df,'Label')


# In[12]:


#Data Encoding
df['Label'] = df['Label'].replace('s',0)
df['Label'] = df['Label'].replace('b',1)


# #### Outlier Treatment

# In[13]:


plt.figure(figsize=(50,50))
num_cols = df.columns
i=1
for col in num_cols:
    plt.subplot(11,3,i)
    sns.boxplot(x=df[col])
    #sns.boxplot(x=boston_df['DIS'])
    i+=1
plt.show()    


# ##### Capping the Outliers

# In[14]:


#Competition Distance
percentiles = df['DER_sum_pt'].quantile([0.01,0.99]).values
df['DER_sum_pt'][df['DER_sum_pt'] <= percentiles[0]] = percentiles[0]
df['DER_sum_pt'][df['DER_sum_pt'] >= percentiles[1]] = percentiles[1]

#DER_pt_h
percentiles = df['DER_pt_h'].quantile([0.01,0.99]).values
df['DER_pt_h'][df['DER_pt_h'] <= percentiles[0]] = percentiles[0]
df['DER_pt_h'][df['DER_pt_h'] >= percentiles[1]] = percentiles[1]

#DER_mass_MMC
percentiles = df['DER_mass_MMC'].quantile([0.01,0.99]).values
df['DER_mass_MMC'][df['DER_mass_MMC'] <= percentiles[0]] = percentiles[0]
df['DER_mass_MMC'][df['DER_mass_MMC'] >= percentiles[1]] = percentiles[1]

#DER_prodeta_jet_jet
percentiles = df['DER_prodeta_jet_jet'].quantile([0.01,0.99]).values
df['DER_prodeta_jet_jet'][df['DER_prodeta_jet_jet'] <= percentiles[0]] = percentiles[0]
df['DER_prodeta_jet_jet'][df['DER_prodeta_jet_jet'] >= percentiles[1]] = percentiles[1]

#PRI_met_sumet
percentiles = df['PRI_met_sumet'].quantile([0.01,0.99]).values
df['PRI_met_sumet'][df['PRI_met_sumet'] <= percentiles[0]] = percentiles[0]
df['PRI_met_sumet'][df['PRI_met_sumet'] >= percentiles[1]] = percentiles[1]

#PRI_met
percentiles = df['PRI_met'].quantile([0.01,0.99]).values
df['PRI_met'][df['PRI_met'] <= percentiles[0]] = percentiles[0]
df['PRI_met'][df['PRI_met'] >= percentiles[1]] = percentiles[1]

#DER_mass_transverse_met_lep
percentiles = df['DER_mass_transverse_met_lep'].quantile([0.01,0.99]).values
df['DER_mass_transverse_met_lep'][df['DER_mass_transverse_met_lep'] <= percentiles[0]] = percentiles[0]
df['DER_mass_transverse_met_lep'][df['DER_mass_transverse_met_lep'] >= percentiles[1]] = percentiles[1]

#DER_deltaeta_jet_jet
percentiles = df['DER_deltaeta_jet_jet'].quantile([0.01,0.99]).values
df['DER_deltaeta_jet_jet'][df['DER_deltaeta_jet_jet'] <= percentiles[0]] = percentiles[0]
df['DER_deltaeta_jet_jet'][df['DER_deltaeta_jet_jet'] >= percentiles[1]] = percentiles[1]

#DER_deltar_tau_lep
percentiles = df['DER_deltar_tau_lep'].quantile([0.01,0.99]).values
df['DER_deltar_tau_lep'][df['DER_deltar_tau_lep'] <= percentiles[0]] = percentiles[0]
df['DER_deltar_tau_lep'][df['DER_deltar_tau_lep'] >= percentiles[1]] = percentiles[1]

#DER_pt_ratio_lep_tau
percentiles = df['DER_pt_ratio_lep_tau'].quantile([0.01,0.99]).values
df['DER_pt_ratio_lep_tau'][df['DER_pt_ratio_lep_tau'] <= percentiles[0]] = percentiles[0]
df['DER_pt_ratio_lep_tau'][df['DER_pt_ratio_lep_tau'] >= percentiles[1]] = percentiles[1]

#PRI_tau_pt
percentiles = df['PRI_tau_pt'].quantile([0.01,0.99]).values
df['PRI_tau_pt'][df['PRI_tau_pt'] <= percentiles[0]] = percentiles[0]
df['PRI_tau_pt'][df['PRI_tau_pt'] >= percentiles[1]] = percentiles[1]

#PRI_lep_pt
percentiles = df['PRI_lep_pt'].quantile([0.01,0.99]).values
df['PRI_lep_pt'][df['PRI_lep_pt'] <= percentiles[0]] = percentiles[0]
df['PRI_lep_pt'][df['PRI_lep_pt'] >= percentiles[1]] = percentiles[1]

#PRI_met
percentiles = df['PRI_met'].quantile([0.01,0.99]).values
df['PRI_met'][df['PRI_met'] <= percentiles[0]] = percentiles[0]
df['PRI_met'][df['PRI_met'] >= percentiles[1]] = percentiles[1]

#PRI_jet_subleading_pt
percentiles = df['PRI_jet_subleading_pt'].quantile([0.01,0.99]).values
df['PRI_jet_subleading_pt'][df['PRI_jet_subleading_pt'] <= percentiles[0]] = percentiles[0]
df['PRI_jet_subleading_pt'][df['PRI_jet_subleading_pt'] >= percentiles[1]] = percentiles[1]

#PRI_jet_leading_pt
percentiles = df['PRI_jet_leading_pt'].quantile([0.01,0.99]).values
df['PRI_jet_leading_pt'][df['PRI_jet_leading_pt'] <= percentiles[0]] = percentiles[0]
df['PRI_jet_leading_pt'][df['PRI_jet_leading_pt'] >= percentiles[1]] = percentiles[1]

#DER_pt_tot
percentiles = df['DER_pt_tot'].quantile([0.01,0.99]).values
df['DER_pt_tot'][df['DER_pt_tot'] <= percentiles[0]] = percentiles[0]
df['DER_pt_tot'][df['DER_pt_tot'] >= percentiles[1]] = percentiles[1]

#DER_mass_jet_jet
percentiles = df['DER_mass_jet_jet'].quantile([0.01,0.99]).values
df['DER_mass_jet_jet'][df['DER_mass_jet_jet'] <= percentiles[0]] = percentiles[0]
df['DER_mass_jet_jet'][df['DER_mass_jet_jet'] >= percentiles[1]] = percentiles[1]

#DER_mass_vis
percentiles = df['DER_mass_vis'].quantile([0.01,0.99]).values
df['DER_mass_vis'][df['DER_mass_vis'] <= percentiles[0]] = percentiles[0]
df['DER_mass_vis'][df['DER_mass_vis'] >= percentiles[1]] = percentiles[1]

#PRI_jet_all_pt
percentiles = df['PRI_jet_all_pt'].quantile([0.01,0.99]).values
df['PRI_jet_all_pt'][df['PRI_jet_all_pt'] <= percentiles[0]] = percentiles[0]
df['PRI_jet_all_pt'][df['PRI_jet_all_pt'] >= percentiles[1]] = percentiles[1]


# In[15]:


plt.figure(figsize=(50,50))
num_cols = df.columns
i=1
for col in num_cols:
    plt.subplot(11,3,i)
    sns.boxplot(df[col])
    i+=1
plt.show()    


# ### Correlation

# In[16]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
plt.show()


# #### Dropping the columns with high correlation value

# In[17]:


df = df.drop(['DER_pt_h','DER_deltaeta_jet_jet','DER_sum_pt','PRI_met_sumet','PRI_jet_num'],1)


# In[18]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[19]:


boson = df['Label'].value_counts().to_frame('counts')
boson


# In[20]:


#Data Imbalance
my_circle = plt.Circle((0,0),0.7,color='white')
plt.pie(boson.counts,labels=['b','s'],colors=['blue','green'],autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# ##### Taking only few columns for model building

# In[21]:


df=df[["DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis",'DER_mass_jet_jet',"DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","Label"]]


# #### Splitting data into predictors(X) and target(y) variable

# In[22]:


X = df.drop(columns=['Label'])
y = df['Label']
X = X.values
y = y.values


# In[23]:


from sklearn.model_selection import train_test_split
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.2)


# In[24]:


#Handling data imbalance with SMOTE function (apply only to train data)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_smote, y_smote = smote.fit_sample(X_train, y_train)


# In[25]:


y_smote = pd.DataFrame(y_smote)
y_smote.describe()


# ##### Scaling the data so that all the values will lie in the same range

# In[26]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_smote = scaler.fit_transform(X_smote)
X_test = scaler.transform(X_test)


# In[27]:


X_smote = pd.DataFrame(X_smote)
y_smote = pd.DataFrame(y_smote)
X_smote.columns = df.columns[:-1]
X_smote.head()


# #### Building Neural Network

# In[28]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[29]:


#Define model
def create_baseline():
    model = Sequential()
    model.add(Dense(30, input_dim=7, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
    return model


# In[30]:


model=create_baseline()


# In[31]:


#Fit the model
history = model.fit(X_smote,y_smote,validation_data=(X_test,y_test),epochs=80)


# In[32]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[33]:


#Model Validation
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[34]:


#Saving the Model
model = tf.keras.models.load_model('Higgs.h5')


# In[ ]:




