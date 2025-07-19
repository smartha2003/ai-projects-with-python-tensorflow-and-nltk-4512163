# Import required modules from NLTK
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random

# Download the movie reviews dataset (if not already downloaded)
nltk.download('movie_reviews')

# 📦 Step 1: Prepare the dataset
# Each document is a tuple: (list of words in review, category ['pos' or 'neg'])
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure a random distribution of positive and negative reviews
random.shuffle(documents)

# 📊 Step 2: Extract features
# Get frequency distribution of all words in the corpus, lowercased
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# Choose the top 2000 most common words as features
word_features = list(all_words)[:2000]

# Feature extractor function
def document_features(document):
    # Convert the list of words in the document to a set for faster lookup
    document_words = set(document)
    features = {}
    # For each of the 2000 most common words, check if it appears in the document
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# 🧠 Step 3: Transform documents into feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# 🏋️ Step 4: Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# 🧪 Step 5: Evaluate the model
print("Accuracy on test set:", accuracy(classifier, test_set))

# 🔍 Step 6: Show top 5 most informative features
classifier.show_most_informative_features(5)