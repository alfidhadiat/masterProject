# 1. Transforms arxiv abstracts into LDA topic distributions
# 2. Using topics to classify categories with linear regression

##########
# Imports
##########

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.feature_selection import f_classif
from time import time
import pandas as pd
import numpy as np
import pickle 
import joblib
import sys

import matplotlib.pyplot as plt

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

def keep_main_cats(df: pd.DataFrame):
    keep_cats = ['cs', 'math', 'cond-mat', 'astro-ph', 'physics']
    df = df[df['first_category'].isin(keep_cats)]
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

def load_lda_topic_coherency():
    with open("lda_coh_per_topics.pkl", "rb") as f:
        lda_topic_coh = pickle.load(f)
    return lda_topic_coh

def crossval_linreg(X: list, y: list, cv: int = 5):
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    return cross_val_score(lr, X, y, cv = cv)

def crossval_dt(X: list, y: list, cv: int = 5):
    dt = DecisionTreeClassifier()
    return cross_val_score(dt, X, y, cv = cv)

def crossval_confusion_linreg(X: list, y: list, cv: int = 5):
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    y_pred = cross_val_predict(lr, X, y, cv = cv)
    return confusion_matrix(y, y_pred)

def train_save_linreg(X: list, y: list, path: str):
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    lr.fit(tf_lda, cats)
    with open(path, 'wb') as f:
        pickle.dump(lr, f)
    print(f"Linear regression saved as {path}")

def load_linreg_model(path: str):
    with open(path, 'rb') as f:
        lr = pickle.load(f)
    return lr

def load_topic_coherences(path: str):
    with open(path, 'rb') as f:
        topic_coh = pickle.load(f)
    return topic_coh

def make_scatterplot(x: list, y: list, label: list):
    plt.scatter(x, y)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.show()

#################
# Main Function
#################

if __name__ == '__main__':

    # Load data
    df = load_arxiv_data("../../dataset/beyond_2018_archive.pkl")
    df.reset_index(inplace = True)
    df = add_first_cats(df)
    df = keep_40k_categories(df)
    cats = grab_first_cats(df)

    # Load pickled BERTopic predictions
    bert_path = '../../classifier_scripts/bertopic_classifiers/predictions/bertopic100_1_probs.pkl'
    probs = pickle.load(open(bert_path, 'rb'))
    docs = [probs[i] for i in df['index']]

    # Load models and coherence scores
    doc_df = pd.DataFrame(docs)

    # Calcuating logistic regression feature importance by single backstep comparison
    base_lr = LogisticRegression(multi_class = 'multinomial', max_iter = 200)
    base_lr.fit(docs, cats)
    base_pred = base_lr.predict_proba(docs)
    base_loss = log_loss(cats, base_pred)
    base_cv_score = np.mean(crossval_linreg(docs, cats))

    topics = []
    loss_diff = []
    cv_diff = []
    for i in range(100):
        cur_topic = doc_df.columns[i]
        cur_df = doc_df.drop(doc_df.columns[i], axis = 1)
        cur_lr = LogisticRegression(multi_class = 'multinomial', max_iter = 200)
        cur_lr.fit(cur_df, cats)
        cur_pred = cur_lr.predict_proba(cur_df)
        cur_loss = log_loss(cats, cur_pred)
        cur_cv_score = np.mean(crossval_linreg(cur_df, cats))
        print('-----')
        print(f"Topic {cur_topic} ")
        print(f"Feat. Imp.: {base_loss - cur_loss}")
        print(f"CV acc. dif.: {base_cv_score - cur_cv_score}")
        topics.append(cur_topic)
        loss_diff.append(base_loss - cur_loss)
        cv_diff.append(base_cv_score - cur_cv_score)

    loss_df = pd.DataFrame(topics)
    loss_df['loss_diff'] = loss_diff
    loss_df['cv_diff'] = cv_diff
    loss_df.columns = ['topics', 'loss_diff', 'cv_diff']
    loss_df = loss_df.sort_values(by = ['loss_diff'], ascending = True)

    loss_df.to_csv('../feature_importances/featcoh_b_lr.csv')

