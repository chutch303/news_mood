

```python
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import consumer_key, consumer_secret, access_token, access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
news_twitters = ['@BBC', '@CBS', '@CNN', '@FoxNews', '@nytimes']
```


```python
#for loop to collect all tweet data from each news org
merged_tweets = []
tweet_count = []

for x in news_twitters:
    num_tweets = 0
    for status in tweepy.Cursor(api.user_timeline, id=x).items(500):
        merged_tweets.append(status)
        num_tweets += 1
        tweet_count.append(num_tweets)
```


```python
df = pd.DataFrame([x._json for x in merged_tweets])[['text', 'created_at', 'user']]
df['label'] = df.user.map(lambda x: x.get('name'))
df['tweet_count'] = tweet_count
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>created_at</th>
      <th>user</th>
      <th>label</th>
      <th>tweet_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RT @BBC6Music: 👏 What was the best gig you saw...</td>
      <td>Sun Jul 01 20:43:50 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RT @bbc5live: “We said jokingly, ‘is it a shar...</td>
      <td>Sun Jul 01 19:36:52 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Get unprecedented access to the hidden world o...</td>
      <td>Sun Jul 01 19:27:02 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>✈️🌍 @RomeshRanga travels way beyond his comfor...</td>
      <td>Sun Jul 01 19:02:03 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Why were these photos of the Great Depression ...</td>
      <td>Sun Jul 01 18:00:19 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
analyzer = SentimentIntensityAnalyzer() # Initialize the class
```


```python
merged_text = [x._json['text'] for x in merged_tweets]
```


```python
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

for tweet in merged_text:
    vs = analyzer.polarity_scores(tweet)
    compound_list.append(vs.get('compound'))
    positive_list.append(vs.get('pos'))
    negative_list.append(vs.get('neg'))
    neutral_list.append(vs.get('neu'))
```


```python
df['compound_score'] = compound_list
df['positive_score'] = positive_list
df['negative_score'] = negative_list
df['neutral_score'] = neutral_list
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>created_at</th>
      <th>user</th>
      <th>label</th>
      <th>tweet_count</th>
      <th>compound_score</th>
      <th>positive_score</th>
      <th>negative_score</th>
      <th>neutral_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RT @BBC6Music: 👏 What was the best gig you saw...</td>
      <td>Sun Jul 01 20:43:50 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>1</td>
      <td>0.8299</td>
      <td>0.290</td>
      <td>0.000</td>
      <td>0.710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RT @bbc5live: “We said jokingly, ‘is it a shar...</td>
      <td>Sun Jul 01 19:36:52 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>2</td>
      <td>0.6588</td>
      <td>0.155</td>
      <td>0.000</td>
      <td>0.845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Get unprecedented access to the hidden world o...</td>
      <td>Sun Jul 01 19:27:02 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>3</td>
      <td>0.4019</td>
      <td>0.130</td>
      <td>0.000</td>
      <td>0.870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>✈️🌍 @RomeshRanga travels way beyond his comfor...</td>
      <td>Sun Jul 01 19:02:03 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>4</td>
      <td>-0.2957</td>
      <td>0.168</td>
      <td>0.176</td>
      <td>0.657</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Why were these photos of the Great Depression ...</td>
      <td>Sun Jul 01 18:00:19 +0000 2018</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>5</td>
      <td>-0.0516</td>
      <td>0.211</td>
      <td>0.273</td>
      <td>0.515</td>
    </tr>
  </tbody>
</table>
</div>




```python
#change 'created at' from sting to datetime objet
from datetime import datetime

def convert_twitter_created_at_to_datetime(string_time):
    return datetime.strptime(string_time,'%a %b %d %H:%M:%S +0000 %Y')

df['created_at'] = df.created_at.map(convert_twitter_created_at_to_datetime)
```


```python
colors = {'BBC':"green", 'CBS':"black", 'CNN':"blue", 'Fox News':"red", 'The New York Times':"yellow"}
df['color'] = df['label'].apply(lambda x: colors[x])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>created_at</th>
      <th>user</th>
      <th>label</th>
      <th>tweet_count</th>
      <th>compound_score</th>
      <th>positive_score</th>
      <th>negative_score</th>
      <th>neutral_score</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RT @BBC6Music: 👏 What was the best gig you saw...</td>
      <td>2018-07-01 20:43:50</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>1</td>
      <td>0.8299</td>
      <td>0.290</td>
      <td>0.000</td>
      <td>0.710</td>
      <td>green</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RT @bbc5live: “We said jokingly, ‘is it a shar...</td>
      <td>2018-07-01 19:36:52</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>2</td>
      <td>0.6588</td>
      <td>0.155</td>
      <td>0.000</td>
      <td>0.845</td>
      <td>green</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Get unprecedented access to the hidden world o...</td>
      <td>2018-07-01 19:27:02</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>3</td>
      <td>0.4019</td>
      <td>0.130</td>
      <td>0.000</td>
      <td>0.870</td>
      <td>green</td>
    </tr>
    <tr>
      <th>3</th>
      <td>✈️🌍 @RomeshRanga travels way beyond his comfor...</td>
      <td>2018-07-01 19:02:03</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>4</td>
      <td>-0.2957</td>
      <td>0.168</td>
      <td>0.176</td>
      <td>0.657</td>
      <td>green</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Why were these photos of the Great Depression ...</td>
      <td>2018-07-01 18:00:19</td>
      <td>{'id': 19701628, 'id_str': '19701628', 'name':...</td>
      <td>BBC</td>
      <td>5</td>
      <td>-0.0516</td>
      <td>0.211</td>
      <td>0.273</td>
      <td>0.515</td>
      <td>green</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('output_data_file.csv')
```


```python
plt.figure(figsize=(10, 10))
plt.scatter(df.tweet_count, df.compound_score, c = df.color)
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.title('Sentiment Analysis of Media Tweets 7/01/18')
plt.savefig('tweet_sentiment_scatter.png')
```


![png](output_11_0.png)



```python
mean_compound = df.groupby(['label']).compound_score.mean().reset_index()
mean_compound
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>compound_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>0.150650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>0.357811</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.017025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox News</td>
      <td>-0.002535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New York Times</td>
      <td>-0.006597</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 5))
plt.bar(mean_compound['label'], mean_compound['compound_score'])
plt.xlabel('News Organizations')
plt.ylabel('Compound Sentiment Score')
plt.title('Sentiment Analysis of Media Tweets 7/01/18')
plt.savefig('tweet_sentiment_bar.png')
```


![png](output_13_0.png)

