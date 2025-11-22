### This file defines a TextProcessor class for text preprocessing tasks such as tokenization, stemming, and lemmatization.
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

# Download required NLTK data
for pkg in ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']:
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

    def get_wordnet_pos(self, treebank_tag):
        """Map NLTK POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to Noun if unsure

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
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words and len(token) > 2]

        # Apply stemming
        stems = [self.porter.stem(token) for token in filtered_tokens]

        # Deduce POS tags and apply lemmatization
        pos_tags = nltk.pos_tag(filtered_tokens)
        lemmas = []
        for token, tag in pos_tags:
            wn_tag = self.get_wordnet_pos(tag)
            lemmas.append(self.lemmatizer.lemmatize(token, pos=wn_tag))

        return {
            'original_text': cleaned_text,
            'tokens': filtered_tokens,
            'stems': stems,
            'lemmas': lemmas,
            'token_count': len(filtered_tokens)
        }
