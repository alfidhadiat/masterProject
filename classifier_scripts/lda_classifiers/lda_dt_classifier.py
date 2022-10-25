# 1. Transforms arxiv abstracts into LDA topic distributions
# 2. Using topics to classify categories with linear regression

##########
# Imports
##########

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from time import time
import pandas as pd
import numpy as np
import argparse
import pickle 
import joblib

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser(description = 'Train a Decision Trees  classifier for the arXiv dataset using LDA features')
parser.add_argument('topic_size', metavar = 't', type = int, help = 'integer: size of topic')
args = parser.parse_args()

#############
# Functions
#############

def add_first_cats(df: pd.DataFrame) -> pd.DataFrame:
    first_cat = []
    print('Adding category preprocessing to dataset...')
    t0 = time()
    for categories in df['category']:
        cats = [category.split('.')[0] for category in categories]
        first_cat.append(cats[0])
    df['first_category'] = first_cat
    print(f"...done in {time() - t0:.2f} seconds")
    return df

def grab_first_cats(df: pd.DataFrame) -> list:
    first_cat = []
    print('Adding category preprocessing to dataset...')
    t0 = time()
    for categories in df['category']:
        cats = [category.split('.')[0] for category in categories]
        first_cat.append(cats[0])
    print(f"...done in {time() - t0:.2f} seconds")
    return first_cat

def keep_40k_categories(df: pd.DataFrame) -> pd.DataFrame:
    if 'first_category' not in df.columns:
        print("First category column does not exist; transform first")
        return
    df = df.groupby('first_category').filter(lambda x: len(x) > 40000)
    d_dict = {}
    print("Balancing documents to 40,000 counts...")
    t0 = time()
    for index, row in df.iterrows():
        cat = row['first_category']
        doc_id = row['id']
        if cat not in d_dict:
            d_dict[cat] = []
        if len(d_dict[cat]) < 40000:
            d_dict[cat].append(doc_id)
    doc_ids = []
    for cat, docs in d_dict.items():
        doc_ids = doc_ids + docs
    df = df[df['id'].isin(doc_ids)]
    print(f"done in {time() - t0:.2f} seconds")
    return df

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

def crossval_dt(X: list, y: list, cv: int = 5):
    dt = DecisionTreeClassifier()
    return cross_val_score(dt, X, y, cv = cv)

#################
# Main Function
#################

if __name__ == '__main__':

    # Argument parsing
    n_topics = args.topic_size
    
    # Load data
    df = load_arxiv_data("../dataset/beyond_2018_archive.pkl")
    cats = grab_first_cats(df)

    # Train and cross validate in loops
    for model_number in range(1, 6):

        # Load models
        model_path = f"../modeling_scripts/lda_models"
        tf_dir = f"lda_models/tf_{n_topics}topics_{model_number}.joblib"
        lda_dir = f"lda_models/lda_{n_topics}topics_{model_number}.joblib"
        tf_path = model_path + tf_dir
        lda_path = model_path + lda_dir
        tf, lda = load_tf_lda_models(tf_path, lda_path)

        # Transform dataset
        tf_abstracts = tf.transform(df['abstract'])
        tf_lda = lda.transform(tf_abstracts)

        # Cross validate 
        scores = crossval_dt(tf_lda, cats)
        mean_score = np.mean(scores)

        # Save results
        with open('lda_classification_results.txt', 'a') as result:
            result.write(f"{n_topics},{model_number},crossval_dt,{mean_score}\n")
