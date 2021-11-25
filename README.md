#Jobs-Unsupervised   

**Clustering job titles by K-Means**   

In order to classify unlabeled textual job titles to  functions in an organization I used  an unsupervised model in the following general stages:   
1. Cleaning the titles – removing punctuations, stop words, doing lemmatization and other minor adaptations.   
2. Vectorization – using sklearn **TfIdf**, ngram_range=(1,2)  
3. Clustering – with **Kmeans**, 7 centers.   

As a representative title for each cluster, I took the most common word in the cluster. Of course they can be converted to parallel organizational function as needed.
The graph projects 50 features on 2 dimensions(by PCA), so it is kind of twisted, but by comparing the representative titles to the titles in the input we can see that there is a pretty good match.   


![image](https://user-images.githubusercontent.com/54791267/143413889-cb6612ef-da6e-4083-a668-dfee085f9f86.png)

![image](https://user-images.githubusercontent.com/54791267/143415919-a44b6625-2211-44b3-96f7-1dfa71615298.png)



