# Import necessary libraries
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel

import nltk
from nltk.tokenize import word_tokenize


def LDA(data, k):

    # LDA works on documents, not on vectors
    string_data = []
    for sublist in data:
        string_data.append(' '.join(map(str, sublist)))

    # Download NLTK Sentence Tokenizer
    nltk.download('punkt')

    # Tokenize the document
    tokens = [word_tokenize(string) for string in string_data]

    # Create a dictionary representation of the document
    dictionary = corpora.Dictionary(tokens)

    # Create a corpus: List of bag-of-words
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=15)

    # Print the topics found by the model
    print(f"Top-{k} latent Semantics for LDA")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
