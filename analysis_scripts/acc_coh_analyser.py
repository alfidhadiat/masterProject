
##########
# Import 
##########

import pandas as pd
import pickle
import sys
from scipy import stats
import sys

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

def load_csv(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        return pd.read_csv(f, sep = ',')

def correlate_coherency_accuracy(tm_df, coh_df):
    coh_meas = ['u_mass', 'c_uci', 'c_npmi', 'c_v']
    for classifier in ['linear_regression', 'decision_trees']:
        print("==================")
        print(classifier)
        for coh in coh_meas:
            ntm_df = tm_df[tm_df['classifier'] == classifier]
            ncoh_df = coh_df[coh_df['coherence_measure'] == coh]

            result_df = pd.merge(ntm_df, ncoh_df, on = ['n_topic', 'n_model'])

            r_val, p_val = stats.spearmanr(result_df['score'], result_df['accuracy'])

            print(coh)
            if p_val < 0.001:
                print(f"{p_val:.4f} ***")
            elif p_val < 0.01:
                print(f"{p_val:.4f} **")
            elif p_val < 0.05:
                print(f"{p_val:.4f} *")
            else:
                print(f"{p_val:.4f} -")

            print(f"{r_val:.4f}")
            print('-----------')

#################
# Main Function
#################

if __name__ == '__main__':

    # Load coherences and accuracies
    lp_df = load_csv('../classifier_scripts/lda_classifiers/lda_class_result.csv')
    bp_df = load_csv('../classifier_scripts/bertopic_classifiers/bert_class_result.csv')
    lc_df = load_csv('../coherency_scripts/lda_coherence_results.csv')
    bc_df = load_csv('../coherency_scripts/bertopic_coherency_results.csv')

    # Calculate all correlations between scores and results
    print('====================')
    print('LDA Correlations')
    correlate_coherency_accuracy(lp_df, lc_df)

    print('====================')
    print('BERTopic Correlations')
    correlate_coherency_accuracy(bp_df, bc_df)
