# Importing the Natural Language Toolkit (NLTK) library
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon required for sentiment analysis
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pre-trained sentiment analysis model
# specifically tuned for social media texts and short sentences
nltk.download('vader_lexicon')

# Define a function to analyze sentiment of input text
def analyze_sentiment(text):
    # Initialize the SentimentIntensityAnalyzer from NLTK
    # It uses a rule-based approach and a pre-defined lexicon to assign sentiment scores
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment scores for the given text
    # Returns a dictionary with 4 scores: negative, neutral, positive, and compound
    sentiment = sia.polarity_scores(text)

    # Print the sentiment scores
    print(sentiment)

# Test the sentiment analyzer with a sample input
# You can change this to experiment with different tones or phrases
analyze_sentiment("NLTK is a great library for Natural Language Processing!")