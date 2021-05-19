---
layout: post
title:  "Topic Modeling using Kmeans"
date:   2021-05-15 17:16:46 +0800
categories: Topic Modeling, Kmeans
---
This post about topic modeling from a collection of twitter data. Topic modeling(主题模型) is a statical model to find a abstract topic from a series text documents, in another words, it helps to understand and summarize large collections of textual information.
In this project, I'll use unsupervised techniques -- K-means to group to identify main topics in the sea of text.

# Prepare
There are couple of third party python library should be installed. First is the [NLTK(Natural Language Toolkit)](http://www.nltk.org/), which could make some operations on textual data such as tokenize, tagging.
Other two libraries were common tools panda and numpy.

# Loading Data
The twitter data was collected as a csv format file, just read it by panda.
![Data snapshoot](/assets/WeChatcd5a291c90f560b8a27fc7ad96a70342.png)

Check if there were exist duplicate tweets, the `df.tweet.nunique()` result was same as the length of data frame, so there was no duplicate tweets at the moment.

# Clean Data
Now let's focused on the 'tweet' column, this was the foundation of topic modeling but there were lots of noise that we didn't care about which even affects our results.
- Remove @ mentions: Mentions someone were not important for topic modeling, just removed them.
```python
df['Clean_text'] = df['tweet'].str.replace("@[\w]*", '')
```
- Remove anything not characters: Numbers or other symbols were not important which we could remove.
```python
df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")
```
- Other: Text case changed, filtered out words length smaller than 2 and drop duplicate blank.
```python
df['Clean_text'] = df['Clean_text'].str.lower()
df['Clean_text'] = df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
```
- Remove duplicate: as one of the final steps, there would exist some duplicate tweets, so checked it.
```python
df.drop_duplicates(subset=['Clean_text'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
```
- Remove empty tweets: the final step, we should find all of the blank 'tweet' rows and remove them.
```python
df['Clean_text_length'] = df['Clean_text'].apply(len)
indexes = df[df['Clean_text_length']==0].index
df.drop(index=indexes, inplace=True)
df.reset_index(drop=True, inplace=True)
```

# Vectorizer
In the sklearn library there are two api functions to convert text data to vectors, say vectorizers, `TfidfVectorizer` and `CountVectorizer`.
`TfidfVectorizer` convert a collection of raw documents to a matrix of TF-IDF features. In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
`CountVectorizer` Convert a collection of text documents to a matrix of token counts. The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
```python
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word',ngram_range=(1,1), stop_words='english', min_df = 0.0001, max_df=0.7)
count_vect.fit(df['Clean_text'])
desc_matrix = count_vect.transform(df["Clean_text"])
```
Checked the vectorizer matrix, it's shaped (19761, 6743).
After this process, the tweets data was prepared for next step to make clusters.

# Words Cluster and Visualization
Cluster operation we relied on `sklearn.cluster.KMeans`, words cloud visualization relied on `wordcloud` library.
Actually if the data was clean and tidy, the rest works were easy, we could just use these third party libraries to do them.
Set the number of clusters, we could get the result easily:
```python
num_clusters = 2
km = KMeans(n_clusters = num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()
```
We could combine the clusters with the tweets as a new data frame:
```python
tweets = {'Tweet': df['Clean_text'].tolist(), 'Cluster':clusters}
new_frame = pd.DataFrame(tweets, index=[clusters])
```

Next step was words cloud visualization, to make it easy to call we wrap it as a function:
```python
def wordcloud(cluster):
  # combining the image with the dataset
  Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

  # We use the ImageColorGenerator library from Wordcloud
  # Here we take the color of the image and impose it over our wordcloud
  image_colors = ImageColorGenerator(Mask)

  # Now we use the WordCloud function from the wordcloud library
  wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(cluster)

  # Size of the image generated
  plt.figure(figsize=(10,20))

  # Here we recolor the words from the dataset to the image's color
  # recolor just recolors the default colors to the image's blue color
  # interpolation is used to smooth the image generated
  plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

  plt.axis('off')
  plt.show()
```
Now we wanted to show the word cloud within cluster 1:
```python
cluster_1 = new_frame[new_frame.Cluster==1]['Tweet']
cluster_1_words = ' '.join(word for word in cluster_1)
wordcloud(cluster_1_words)
```
This word cloud shown as like below:
![cluster 1 when the number of cluster was 2](/assets/WeChat816f864d2664d37647b0238e72773fbe.png)


That's it, it looked good. Work almost done here, but the effectiveness of kmeans algorithm largely depends on initial cluster number, that's say you want to set data into 2 or 6 clusters. It deserved to make more experiments to compare the results. In order to make it easier, we wrap this process as a function:
```python
def identify_topics(df, desc_matrix, num_clusters):
  km = KMeans(n_clusters=num_clusters)
    km.fit(desc_matrix)
    clusters = km.labels_.tolist()
    tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
    frame = pd.DataFrame(tweets, index = [clusters])
    print(frame['Cluster'].value_counts())

    for cluster in range(num_clusters):
        cluster_words = ' '.join(text for text in frame[frame['Cluster'] == cluster]['Tweet'])
        wordcloud(cluster_words)
```
