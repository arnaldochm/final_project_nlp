import streamlit as st
from pickle import load
import streamlit as st
import re
import pandas as pd
import numpy as np
from scipy import stats
import nltk
from nltk import download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from textstat import flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

model = load(open("models/tunedXGboostRegressor.pkl", "rb"))
# Load PCA Model
with open('models/pca_model.pkl', 'rb') as f:
    pca = load(f)
# Load the pickled Text Vectorizer model
with open('models/text_vectorizer_model.pkl', 'rb') as f:
    tfidfvectorizer_model = load(f)


st.title("Predicting Amazon Book Review Scores")

download("wordnet", quiet=True)
lemmatizer = WordNetLemmatizer()

# Download Stop Words
download("stopwords", quiet=True)
stop_words = stopwords.words("english")

#SENTIMENT
# Download Vader Lexicon
nltk.download('vader_lexicon', quiet=True)


fiction_category = 1
top_publisher = 1

helfulness = st.slider("helfulness", min_value = 0.00, max_value = 1.00, step = 0.01)
str_category = st.selectbox("Choose a Category:", ['Fiction', 'Juvenile Fiction', 'Religion', 'Biography & Autobiography',
       'History', 'Business & Economics'])
str_publisher = st.selectbox("Choose a Publisher:", ['Penguin', 'Harper Collins', 'Simon and Schuster', 'Basic Health Publications', 'Basic Health Publications', 'Nbm Publishing Company'])
reviewText = st.text_input("Review Text")

#Binarize category
category = 0
if str_category in ['Fiction', 'Juvenile Fiction']:
    category = 1

publisher = 0
if str_publisher in ['Penguin', 'Harper Collins', 'Simon and Schuster']:
    publisher = 1

# Text To Lower
clean_text = reviewText.lower()
# Extract special characters and numbers
clean_text = re.sub(r'[^a-z]', ' ', clean_text)
# Extract numbers
# Change multiple white spaces to a single white space
clean_text = re.sub(r'\s+', ' ',clean_text)

#Lemmatize Text and removing Stopwords
def lemmatize_text(words, lemmatizer = lemmatizer):
    words = words.split(' ')
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2] #Keep Words with more than 3 letters
    return ' '.join(tokens)

clean_text = lemmatize_text(clean_text)


#CALC Text Complexity
#Download Readability test library
nltk.download('punkt', quiet=True)

if len(reviewText) > 0 and reviewText!='':

    def calculate_complexity(review):
        return flesch_kincaid_grade(review)

    text_complexity = calculate_complexity(reviewText)

    vaderSentimentAnalyzer = SentimentIntensityAnalyzer()

    scores = vaderSentimentAnalyzer.polarity_scores(str(clean_text))

    compound_sentiment_original = scores['compound']
    compound_sentiment = np.arctanh(compound_sentiment_original)

    #GET WORD COUNT
    # Function to count words in a text column
    def count_words(text):
        tokens = word_tokenize(text)  # Tokenize the text into words
        return len(tokens)

    word_count = count_words(clean_text)
    word_count = stats.boxcox(word_count, -0.15804217863750494)

    # Preprocessing for text data
    # Create a DataFrame with a single column
    df = pd.DataFrame({'text_clean': [clean_text]})
    text_features_test = tfidfvectorizer_model.transform(df['text_clean'])

    text_features_test_pcaed = pca.transform(text_features_test.toarray())
    numerical_features_nonscaled_test = [[helfulness, compound_sentiment, text_complexity, word_count, category, publisher]]

    X_test = np.concatenate((text_features_test_pcaed, numerical_features_nonscaled_test), axis=1)

    if st.button("Predict"):
        prediction = str(model.predict(X_test)[0]) 
        st.write("Predicted Review Score:", prediction)
        st.write("Text Complexity:", text_complexity)
        st.write("Sentiment:", compound_sentiment_original)