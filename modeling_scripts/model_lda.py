
##########
# Imports
##########

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import argparse
import pandas as pd
import pickle 
import joblib

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser(description = 'Train LDA model with specific topic size; identify with model number')
parser.add_argument('topic_size', metavar = 't', type = int, help = 'integer: size of topic')
parser.add_argument('model_number', metavar = 'n', type = int, help = 'integer: model number')
args = parser.parse_args()

############
# Grab data
############

with open("../dataset/beyond_2018_archive.pkl", "rb") as f:
    file = pickle.load(f)

df = pd.DataFrame(file)

###############
# LDA and count vectorizer function 
###############

n_features = args.topic_size
model_number = args.model_number

tf_vectorizer = CountVectorizer(
        max_df = 0.95, min_df = 2, stop_words = "english")
tf = tf_vectorizer.fit_transform(df["abstract"])

lda = LatentDirichletAllocation(n_components = n_features, verbose = 1)
lda.fit(tf)

###############
# Save model
###############

tf_path = f"lda_models/tf_{n_features}topics_{model_number}.joblib"
lda_path = f"lda_models/lda_{n_features}topics_{model_number}.joblib"

joblib.dump(tf_vectorizer, tf_path)
joblib.dump(lda, lda_path)
