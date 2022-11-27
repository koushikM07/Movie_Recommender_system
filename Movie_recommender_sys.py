#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd


# In[60]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[61]:


movies.head()


# In[62]:


credits.head()


# In[63]:


movies = movies.merge(credits,on='title')


# In[64]:


movies.shape


# In[65]:


movies['original_language'].value_counts()


# In[66]:


movies.info()


# In[67]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[68]:


movies.head()


# In[69]:


movies.isnull().sum()


# In[70]:


movies.dropna(inplace=True)


# In[71]:


movies.duplicated().sum()


# In[72]:


movies.iloc[0].genres


# In[73]:


import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[74]:


movies['genres'] = movies['genres'].apply(convert)


# In[75]:


movies.head()


# In[76]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[77]:


movies.head()


# In[78]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[79]:


movies['cast']=movies['cast'].apply(convert3)


# In[80]:


movies.head()


# In[81]:



def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director' :
            L.append(i['name'])
            break
    return L


# In[82]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[83]:


movies.head()


# In[84]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[85]:


movies.head()


# In[86]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[87]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[88]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[89]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[90]:


movies.head()


# In[91]:


movies['tags']= movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[92]:


movies.head()


# In[93]:


new_df=movies[['movie_id','title','tags']]


# In[94]:


new_df.head()


# In[95]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[96]:


new_df


# In[98]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[99]:


new_df.head()


# In[114]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[115]:


vectors= cv.fit_transform(new_df['tags']).toarray()


# In[116]:


vectors


# In[117]:


cv.get_feature_names()


# In[105]:


import nltk


# In[106]:


from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()


# In[111]:


def stem(text):
    y=[]
     
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[113]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[118]:


from sklearn.metrics.pairwise import cosine_similarity


# In[120]:


similarity = cosine_similarity(vectors)


# In[125]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances =similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[126]:


recommend('Avatar')


# In[127]:


import pickle


# In[128]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[129]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[130]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




