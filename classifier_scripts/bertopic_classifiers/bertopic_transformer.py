
###########
# Imports
###########

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import pandas as pd
import argparse
import joblib
import pickle
import sys

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser(description = 'Train a Decision Trees  classifier for the arXiv dataset using LDA features')
parser.add_argument('topic_size', metavar = 't', type = int, help = 'integer: size of topic')
parser.add_argument('model_number', metavar = 'n', type = int, help = 'integer: model number')
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
    print('-----------')
    return df

def grab_first_cats(df: pd.DataFrame) -> list:
    first_cat = []
    print('Adding category preprocessing to dataset...')
    t0 = time()
    for categories in df['category']:
        cats = [category.split('.')[0] for category in categories]
        first_cat.append(cats[0])
    print(f"...done in {time() - t0:.2f} seconds")
    print('-----------')
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
    print('-----------')
    return df

def load_arxiv_data(path: str, amount: int = None) -> pd.DataFrame:
    with open(path, 'rb') as f:
        df = pd.DataFrame(pickle.load(f))
    if amount:
        df = df[0:amount]
    return df

def load_tf_models(tf_path: str):
    tf = joblib.load(tf_path)
    return tf

def crossval_linreg(X: list, y: list, cv: int = 5):
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    return cross_val_score(lr, X, y, cv = cv)

def crossval_confusion_linreg(X: list, y: list, cv: int = 5):
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    y_pred = cross_val_predict(lr, X, y, cv = cv)
    return confusion_matrix(y, y_pred)

#################
# Main Function
#################

if __name__ == '__main__':

    # Model labels
    n_topics = args.topic_size
    model_number = args.model_number

    # Model paths
    models_path = '../../modeling_scripts/bertopic_models'
    model_path = f"/bertopic{n_topics}_{model_number}"
    model = models_path + model_path
    
    # Load data
    df = load_arxiv_data("../../dataset/beyond_2018_archive.pkl")
    cats = grab_first_cats(df)

    # Load models
    print("Loading BERTopic model...")
    t0 = time()
    bertopic = BERTopic.load(model)
    print(f"done in {time() - t0:.2f} seconds")
    print('-----------')

    # Predicting papers
    print("BERTopic predicting...")
    t0 = time()
    pred, probs = bertopic.transform(df['abstract'])
    print(f"done in {time() - t0:.2f} seconds")
    print('-----------')

    pred_dir = f"predictions"
    doc_dir = f"/bertopic{n_topics}_{model_number}_docs.pkl"
    prob_dir = f"/bertopic{n_topics}_{model_number}_probs.pkl"
    doc_path = pred_dir + doc_dir
    prob_path = pred_dir + prob_dir

    with open(doc_path, 'wb') as f:
        pickle.dump(pred, f)

    with open(prob_path, 'wb') as f:
        pickle.dump(probs, f)
