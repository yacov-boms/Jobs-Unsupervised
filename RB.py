# Main
# import packages
from IntClean import clean_example
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Read the combined file
df = pd.read_excel(r'C:\Users\Administrator\Desktop\RB.xlsx', sheet_name='Worksheet')
df_big = df

df = df_big.sample(n=300000, random_state=1) # Take a sample
titles = df['title'].to_numpy()
titles = [clean_example(str(ex).lower()) for ex in titles]  # Clean List of queries

# Vectorize
max_features = 50
vectorizer = TfidfVectorizer(tokenizer=word_tokenize,ngram_range=(1,2), 
                             binary=True, max_features=max_features)
X=vectorizer.fit_transform(titles)

# Cluster with Kmeans
centers = 7
kmeans = KMeans(n_clusters=centers).fit(X)
cluster_numbers = kmeans.predict(vectorizer.transform(titles))

# Construct df containing cluster numbers
df_titles = pd.DataFrame(list(zip(titles, cluster_numbers)), columns=['Titles', 'Clusters'])
labels = df_titles['Clusters']
# cluster distribution
df_titles['Clusters'].value_counts()
# 0    103275
# 2     77855
# 4     39480
# 3     23502
# 6     21569
# 1     17530
# 5     16789

# Graphical representation of points
# Project all features on 2 dimensions
pca = PCA(n_components = max_features).fit(X.toarray())
X_pca = pca.transform(X.toarray())
centers_PC = pca.transform(kmeans.cluster_centers_)
sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=cluster_numbers, palette="Set2")
plt.scatter(x=centers_PC[:,0],y=centers_PC[:,1],s=200,c="k",marker="X")

# As a representative title for each cluster, take the most common word in the cluster
def best_feature(documents):
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize,ngram_range=(1,2), 
                                 binary=True, max_features=max_features)
    tfidf = vectorizer.fit_transform(documents)
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(vectorizer.get_feature_names())
    return str(tfidf_feature_names[importance[0]])
# Construct a list of representative word for each cluster
best_features = []
for i in range(centers):
    documents = df_titles[df_titles.Clusters == i].Titles
    best_features.append(best_feature(documents))
# add the representative word to each job title    
df_titles['Function'] = df_titles['Clusters'].apply(lambda x: best_features[x])    



