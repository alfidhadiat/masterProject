# 1. Transforms arxiv abstracts into LDA topic distributions
# 2. Calculate coherency of LDA model

##########
# Imports
##########

from sklearn.linear_model import LogisticRegression
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import pandas as pd
import numpy as np
import argparse
import pickle 
import joblib

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

def load_tf_lda_models(tf_path: str, lda_path: str):
    tf = joblib.load(tf_path)
    lda = joblib.load(lda_path)
    return tf, lda

#################
# Main Function
#################

#    Explanation:
#    metric_coherence_gensim from tmtoolkit requires three things:
#        1. topic_word_distrib: the topic word distribution of the model 
#        2. dtm: the document-term frequency
#        3. texts: the text/documents tokenized

if __name__ == '__main__':

    # Presets
    n_topics = args.topic_size
    coh_meas = ['u_mass', 'c_uci', 'c_npmi', 'c_v']
    
    # Load data
    df = load_arxiv_data("/data/s2863685/beyond_2018_archive.pkl")

    # Load models
    for coh in coh_meas:
        for model_number in range(1, 6):
            model_dir = f"/data/s2863685/remodelers/lda/lda_models/topic_{n_topics}"
            tf_dir = f"/tf_{n_topics}topics_{model_number}.joblib"
            lda_dir = f"/lda_{n_topics}topics_{model_number}.joblib"
            tf_path = model_dir + tf_dir
            lda_path = model_dir + lda_dir
            tf, lda = load_tf_lda_models(tf_path, lda_path)

            # Transform dataset
            tf_abstracts = tf.transform(df["abstract"])
            tf_lda = lda.transform(tf_abstracts)
            tokenizer = tf.build_tokenizer()
            token_text = [tokenizer(doc) for doc in df['abstract']]

            # Calculate coherency
            coherency = metric_coherence_gensim( measure = coh, 
                top_n = 5, 
                topic_word_distrib = lda.components_, 
                dtm = tf_abstracts, 
                vocab = np.array(tf.get_feature_names_out()), 
                texts = token_text)
            mean_coh = np.mean(coherency)
            
            with open("lda_coherence_results.txt", "a") as results:
                results.write(f"{n_topics},{model_number},{coh},{mean_coh}\n")

            topic_coh_dir = f"lda_coherencies"
            topic_dir = f"/l{n_topics}_{model_number}_{coh}.pkl"
            tc_path = topic_coh_dir + topic_dir
            with open(tc_path, 'wb') as f:
                pickle.dump(coherency, f)


