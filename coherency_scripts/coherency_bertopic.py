
###########
# Imports
###########

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import pandas as pd
import numpy as np
import joblib
import pickle
import argparse

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser(description = 'Compute coherency of LDA models with specific topic size')
parser.add_argument('topic_size', metavar = 't', type = int, help = 'integer: size of topic')
args = parser.parse_args()

#############
# Functions
#############

def load_arxiv_data(path: str, amount: int = None) -> pd.DataFrame:
    with open(path, 'rb') as f:
        df = pd.DataFrame(pickle.load(f))
    if amount:
        df = df[0:amount]
    return df

#################
# Main Function
#################

if __name__ == '__main__':

    # Presets
    n_topics = args.topic_size
    coherence_measures = ['u_mass', 'c_uci', 'c_npmi', 'c_v']

    # Load data
    df = load_arxiv_data("/data/s2863685/beyond_2018_archive.pkl")

    # Loop through each measure and model number
    for measure in coherence_measures:
        for model_number in range(1, 6):

            # Load models
            bert_dir = f"/data/s2863685/remodelers/bertopic/bertopic_models/topic_{n_topics}"
            model_dir = f"/bertopic{n_topics}_{model_number}"
            model_path = bert_dir + model_dir
            bertopic = BERTopic.load(model_path)

            # Preprocessing
            documents = df['abstract']
            cleaned_documents = bertopic._preprocess_text(documents)
            vectorizer_ = bertopic.vectorizer_model
            tokenizer_ = vectorizer_.build_tokenizer()
            words = vectorizer_.get_feature_names()
            tokens = [tokenizer_(doc) for doc in cleaned_documents]
            dictionary = corpora.Dictionary(tokens)
            corpus = [dictionary.doc2bow(token) for token in tokens]
            topic_words = [[tup[0] for tup in bertopic.get_topic(topic)] for topic in range(len(set(bertopic.get_topics()))-1)]

            # Coherence measuring
            coherence_model = CoherenceModel(
                    topics = topic_words,
                    texts = tokens,
                    corpus = corpus,
                    dictionary = dictionary,
                    coherence = measure,
                    topn = 5
                    )

            topics_coh = coherence_model.get_coherence_per_topic()
            mean_coh = np.mean(topics_coh)

            # Record model coherence and keep topic coherences
            with open('bertopic_coherency_results.txt', 'a') as results:
                results.write(f"{n_topics},{model_number},{measure},{mean_coh}\n")

            topic_coh_dir = f"/data/s2863685/coherency/bertopic/seperate_coherency/b{n_topics}_coh"
            topic_dir = f"/b{n_topics}_{model_number}_{measure}.pkl"
            tc_path = topic_coh_dir + topic_dir
            with open(tc_path, 'wb') as f:
                pickle.dump(topics_coh, f)
