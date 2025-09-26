# Twitter Sentiment Analysis (Live Tweets)

This project fetches live tweets using the Twitter API and performs sentiment analysis
(positive/negative) using Logistic Regression and TF-IDF.

##  Features

- Fetches live tweets from Twitter API (v2)
- Preprocesses tweets (cleaning, removing links/hashtags)
- Trains a Logistic Regression sentiment model
- Predicts sentiment of live tweets
- Saves results into a CSV file

##  Tech Stack

- Python 3
- scikit-learn
- pandas, numpy, matplotlib
- Twitter API v2
- python-dotenv (for API key security)

##  Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/Sammihan/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Create a `.env` file and add your Twitter Bearer Token:

3. Install dependencies:

```bash
pip install -r requirements.txt

4. Run this script
python main.py

```


