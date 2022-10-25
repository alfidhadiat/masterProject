##########
# Imports
##########

import pandas as pd
import dask.bag as db
import json, pickle

################################
# Parse Arxiv Metadata Snapshot
################################

path = "arxiv-metadata-oai-snapshot.json"
docs = db.read_text(path).map(json.loads)

get_latest_version = lambda x: x['versions'][-1]['created']
trim = lambda x: {'id': x['id'],
                  'authors': x['authors'],
                  'title': x['title'],
                  'doi': x['doi'],
                  'category': x['categories'].split(' '),
                  'abstract': x['abstract'],}
columns = ['id', 'category', 'abstract']

if __name__ == "__main__":
    docs_df = (docs.filter(lambda x: int(get_latest_version(x).split(' ')[3]) > 2018).map(trim).compute())
    with open("beyond_2018_archive.pkl", "wb") as f:
        pickle.dump(docs_df, f)

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

########
# Main
########

if __name__ == "__main__":
    docs_df = (docs.filter(lambda x: int(get_latest_version(x).split(' ')[3]) > 2018).map(trim).compute())
    docs_df = add_first_cats(docs_df)
    docs_df = keep_40k_categories(docs_df)
    with open("beyond_2018_archive.pkl", "wb") as f:
        pickle.dump(docs_df, f)
