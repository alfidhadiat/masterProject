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

