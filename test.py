import nltk
from textblob import TextBlob

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('conll2000')
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('universal_tagset')
nltk.download('tagsets')
nltk.download('brown')
nltk.download('wordnet_ic')

nltk.download('punkt')  # Ensure punkt is available
print(TextBlob("Test sentence.").words)  # Should return ['Test', 'sentence']
