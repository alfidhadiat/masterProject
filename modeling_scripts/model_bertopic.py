
###########
# Imports
###########

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import pandas as pd
import pickle
import sys

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser(description = 'Train LDA model with specific topic size; identify with model number')
parser.add_argument('topic_size', metavar = 't', type = int, help = 'integer: size of topic')
parser.add_argument('model_number', metavar = 'n', type = int, help = 'integer: model number')
args = parser.parse_args()

#############
# Grab Data
#############

with open("../dataset/beyond_2018_archive.pkl", "rb") as f:
    df = pd.DataFrame(pickle.load(f))

############################
# Count vectorizer function 
############################

tf_vectorizer = CountVectorizer(
        max_df = 0.95, min_df = 2, stop_words = "english")
tf = tf_vectorizer.fit_transform(df["abstract"])

#################
# Train BERTopic
#################

n_topics = args.topic_size
model_number = args.model_number

model = BERTopic(
        vectorizer_model = tf_vectorizer,
        language = 'english',
        verbose = True,
        calculate_probabilities = True, 
        min_topic_size = 50,
	nr_topics = n_topics)

model.fit(df['abstract'])

model_name = f"bertopic_models/bertopic{n_topics}_{model_number}"
model.save(model_name)

