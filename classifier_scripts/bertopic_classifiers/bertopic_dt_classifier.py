# 1. Load bertopic arxiv predictions computed in Peregrine
# 2. Using topics to classify categories with linear regression

##########
# Imports
##########

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from time import time
import pandas as pd
import numpy as np
import argparse
import pickle 
import joblib
import sys

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

    # Model Settings
    n_topic = args.topic_size

    # Load data
    df = load_arxiv_data("../../dataset/beyond_2018_archive.pkl")
    df.reset_index(inplace = True)
    cats = grab_first_cats(df)
    
    # Classify through each BERTopic prediction
    for n_model in range(1, 6):

        # Load BERTopic probabilities
        model_path = 'predictions'
        prob_path = model_path + f"/bertopic{n_topic}_{n_model}_probs.pkl"
        bert_probs = pickle.load(open(prob_path, 'rb'))
        doc_probs = [bert_probs[i] for i in df['index']]

        # Cross validate linear regression
        scores = crossval_dt(doc_probs, cats)
        mean_score = np.mean(scores)

        # Save results
        with open('bertopic_class_results.txt', 'a') as result:
            result.write(f"{n_topic},{n_model},linear_regression,{mean_score}\n")
