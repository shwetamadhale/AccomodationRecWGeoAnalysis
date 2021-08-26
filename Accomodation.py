#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#data cleaning


# In[3]:


df = pd.read_csv("food_coded.csv")


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


features = df[['cook', 'eating_out', 'employment', 'ethnic_food', 'exercise', 'fruit_day','income', 'on_off_campus', 'pay_meal_out', 'sports', 'veggies_day']]


# In[7]:


features.dropna(axis = 0, inplace = True)
features.head(10)


# In[8]:


#data exploration and visualization


# In[9]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[10]:


#box plot for the df
sns.set(rc={'figure.figsize':(18,8)})
ax = sns.boxplot(data = df[features])


# In[ ]:


#running k means clustering on the data
#group locn based on ameneties arounf
#location with more shops=Amenity Rich, loation with sparse shops=Amenity poor
#finding the best value for k


# In[11]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


# In[12]:


x = df.iloc[:, [10,19,20,21,22,30,37,46,48,52,57]].values


# In[13]:


kclusters = 2 #setting no of clusters


# In[14]:


#running K Means Clustering
kmeans = KMeans(n_clusters = kclusters, random_state = 0).fit(features)
features['Cluster'] = kmeans.labels_


# In[15]:


fig , axes = plt.subplots(1, kclusters, figsize = (18,8), sharey = True)
axes[0].set_ylabel('Values')

for k in range(kclusters):
    plt.sca(axes[k])
    plt.xticks(rotation = 45, ha = 'right')
    sns.boxplot(data = features[features['Cluster'] == k].drop('Cluster',1), ax = axes[k])
    
plt.show()


# In[16]:


kclusters = 3 #setting no of clusters


# In[17]:


#running K Means Clustering
kmeans = KMeans(n_clusters = kclusters, random_state = 0).fit(features)
features['Cluster'] = kmeans.labels_


# In[18]:


fig , axes = plt.subplots(1, kclusters, figsize = (18,8), sharey = True)
axes[0].set_ylabel('Values')

for k in range(kclusters):
    plt.sca(axes[k])
    plt.xticks(rotation = 45, ha = 'right')
    sns.boxplot(data = features[features['Cluster'] == k].drop('Cluster',1), ax = axes[k])
    
plt.show()


# In[19]:


# insights
# 3 clusters: one high income---> healthier food while others are more lax


# In[20]:


#possible conclusions
#high income students-----> stay on campus/ not cook often/ eat out more
#one subset of this------->est out less often/ eat ethnic food, fruits, veggies
#lower income students---->stay off campus/ likely to cook/ eat more fruits and veggies/ pay less for a meal out


# In[21]:


#best k value = 3


# In[22]:


#get geological data using FOURSQUARE API to find accomodations


# In[23]:


#defining parameters for query


# In[24]:


search_query = 'Apartment' #look for residential locations
radius = 18000 #18km of radius
#college location
cllg_latitude = 19.04300564305559
cllg_longitude =  73.02302253936172


# In[25]:


CLIENT_ID = 'QC3GXEXNVYGQIE4TU0MSGVSCHDPIFRHELEDVS0ACR1QVB51S'
CLIENT_SECRET = 'CINFPIPJ0VH3WKTOCKFXW04OPTLNWRKRYAM2CQEGYQ4EIWM1'
VERSION = '20210824'
LIMIT = 200


# In[26]:


url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    cllg_latitude, 
    cllg_longitude, 
    VERSION, 
    search_query, 
    radius, 
    LIMIT)


# In[27]:


#store results of query 


# In[28]:


from IPython.display import Image 


# In[29]:


from IPython.core.display import HTML 


# In[30]:


import simplejson as json


# In[31]:


from pandas.io.json import json_normalize


# In[32]:


import folium


# In[36]:


import requests


# In[37]:


results = requests.get(url).json()


# In[38]:


venues = results['response']['venues']


# In[40]:


dataframe = json_normalize(venues)
dataframe.head()


# In[ ]:


#removing redundant columns, data filtering


# In[41]:


fil_cols = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location')]+['id']
dataframe_fil = dataframe.loc[:,fil_cols]


# In[42]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[43]:


#filtering category row-wise
dataframe_fil['categories'] = dataframe_fil.apply(get_category_type, axis = 1)


# In[ ]:


#data cleaning


# In[ ]:


#just leaving last term for col names


# In[46]:


dataframe_fil.columns = [column.split('.')[-1] for column in dataframe_fil.columns]
dataframe_fil.drop(['cc','country','state','city','formattedAddress'],axis = 1,inplace = True) 
dataframe_fil.head()


# In[ ]:


#evaluating ideal locations for studemts


# In[51]:


df_evaluate = dataframe_fil[['lat','lng']]


# In[60]:


RestList=[]
latitudes = list(dataframe_fil.lat)
longitudes = list( dataframe_fil.lng)
for lat, lng in zip(latitudes, longitudes):    
    radius = 5000 
    latitude = lat #Query for the apartment location in question
    longitude = lng
    url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
    search_query = 'Restaurant' #searching food related locn
    results = requests.get(url).json()
    #give relevant data for venues in json
    venues = results['response']['venues']
    #change venues to dataframe
    dataframe2 = pd.json_normalize(venues)
    fil_cols = ['name', 'categories'] + [col for col in dataframe2.columns if col.startswith('location.')] + ['id']
    dataframe_fil2 = dataframe2.loc[:, fil_cols]
    # filtering ccategory row=wise
    dataframe_fil2['categories'] = dataframe_fil2.apply(get_category_type, axis = 1)
    # clean column names by keeping only last term
    dataframe_fil2.columns = [column.split('.')[-1] for column in dataframe_fil2.columns]
    RestList.append(dataframe_fil2['categories'].count())


# In[61]:


df_evaluate['Restaurants'] = RestList


# In[62]:


kclusters = 3

# run k-means clustering
kmeans = KMeans(n_clusters = kclusters, random_state = 0).fit(df_evaluate)
df_evaluate['Cluster']=kmeans.labels_
df_evaluate['Cluster']=df_evaluate['Cluster'].apply(str)
df_evaluate.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




