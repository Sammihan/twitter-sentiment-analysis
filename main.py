import requests
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv
load_dotenv()

# ==============================
# 1. Twitter API Setup
# ==============================
BEARER_TOKEN = os.getenv("BEARER_TOKEN")   
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

def get_tweets(query, max_results=20):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {"query": query, "max_results": max_results, "tweet.fields": "lang"}
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} {response.text}")
    
    data = response.json()
    tweets = []
    for t in data.get("data", []):    
        if t.get("lang") == "en":    
            tweets.append(t["text"]) 

    return tweets


# ==============================
# 2. Preprocessing
# ==============================
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)  
    text = re.sub(r"[^a-z\s]", "", text)  
    return text.strip()


# ==============================
# 3. Dummy Sentiment Dataset (for training)
# ==============================
# You can replace this with a real dataset (like Kaggle Sentiment140)
train_texts = [
    "i love this movie", "this is a great day", "happy with the results",
    "i hate this place", "this is the worst ever", "i am very sad"
]
train_labels = [1, 1, 1, 0, 0, 0]  # 1=positive, 0=negative

# Clean the training data
cleaned_train_texts = []          

for t in train_texts:             
    cleaned_text = clean_tweet(t) 
    cleaned_train_texts.append(cleaned_text)  

train_texts = cleaned_train_texts  



# ==============================
# 4. Train Model
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_texts)
y = train_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

print("✅ Training done")
print("🔹 Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# ==============================
# 5. Fetch Live Tweets & Predict
# ==============================
topic = "AI"   # <-- change topic here
tweets = get_tweets(topic, max_results=10)

cleaned_tweets = []  # create an empty list to store cleaned tweets

for t in tweets:                     
    cleaned_text = clean_tweet(t)    
    cleaned_tweets.append(cleaned_text)  

X_new = vectorizer.transform(cleaned_tweets)
preds = model.predict(X_new)

# Put results in DataFrame
df = pd.DataFrame({"tweet": tweets, "cleaned": cleaned_tweets, "sentiment": preds})
df["sentiment"] = df["sentiment"].map({1: "Positive", 0: "Negative"})

print("\n🔹 Live Sentiment Results on topic:", topic)
print(df)

# Save to CSV
df.to_csv("twitter_sentiment_results.csv", index=False)
print("\n📁 Results saved to twitter_sentiment_results.csv")
