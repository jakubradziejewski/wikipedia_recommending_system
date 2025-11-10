import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
for pkg in ['punkt', 'wordnet', 'stopwords']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass



class TextProcessor:
    """Handles text preprocessing including tokenization, stemming, and lemmatization"""

    def __init__(self):
        self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Remove unwanted characters and normalize text"""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_text(self, text):
        """Process text: tokenize, remove stopwords, and apply stemming/lemmatization"""
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text.lower())

        # Filter: keep only alphabetic tokens that are not stopwords
        filtered_tokens = [
            token for token in tokens
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]

        # Apply stemming
        stems = [self.porter.stem(token) for token in filtered_tokens]

        # Apply lemmatization
        lemmas = [self.lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]

        return {
            'original_text': cleaned_text,
            'tokens': filtered_tokens,
            'stems': stems,
            'lemmas': lemmas,
            'token_count': len(filtered_tokens)
        }
