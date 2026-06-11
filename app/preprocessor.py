import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data at module load
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """
    A comprehensive text preprocessor for NLP tasks.
    Supports lowercasing, punctuation removal, URL stripping,
    tokenization, stopword removal, and lemmatization.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def strip_urls(self, text: str) -> str:
        """Removes URLs from text."""
        return re.sub(r'http[s]?://\S+|www\.\S+', '', text)

    def lowercase(self, text: str) -> str:
        """Converts text to lowercase."""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Removes punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str) -> list:
        """Tokenizes text using NLTK."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Removes stopwords from a list of tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        """Lemmatizes a list of tokens."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self, text: str) -> str:
        """
        Chains all preprocessing steps together.
        Returns the processed text as a single string.
        """
        if not isinstance(text, str):
            return ""
            
        text = self.strip_urls(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        
        return " ".join(tokens)

# Global instance for easy importing
preprocessor = TextPreprocessor()
