import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources silently
for pkg in ['punkt', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)


def clean_text(text: str) -> str:
    """Basic Wikipedia text cleanup."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text(text: str):
    """Tokenize, remove stopwords, stem, and lemmatize."""
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    text = clean_text(text)
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in stop_words]

    stems = [ps.stem(t) for t in filtered]
    lemmas = [lemmatizer.lemmatize(t, pos='v') for t in filtered]

    return {
        "original": text,
        "tokens": filtered,
        "stems": stems,
        "lemmas": lemmas,
        "token_count": len(filtered)
    }
