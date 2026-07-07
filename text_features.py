import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

"""
Steps for text preprocessing:
1. Tokenization: word-level tokenization. 
2. Stopwords removal.
3. Lemmatization. 
4. Vectorization with TF-IDF. 
"""


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# tokenize split sentence into smaller chunks: word-tokenization
def tokenize(text: str):

    text = text.lower()
    return word_tokenize(text)

# stopwords are articles, pronouns, prepositions, conjuctions
def remove_stopwords(tokens: list[str]):
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens

# lemmatization -> reducing a word: "running" = "run," "better" = "good,"
def lemmatize(tokens: list[str]):

    lemmatized_tokens = []
    for token in tokens:
        lemmatized_token = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens


def preprocess_text(text: str):

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def vectorize_tfidf(train_texts: list[str], max_features: int = 10000, ngram_range: tuple = (1, 2)):

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    train_vectors = vectorizer.fit_transform(train_texts)
    return vectorizer, train_vectors