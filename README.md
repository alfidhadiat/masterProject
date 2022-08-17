# Master Project for Computational Cognitive Science MSc.

This project is for the purposes of completing the Computational Cognitive Science Masters. The purpose of the project is to correlate topic model coherency with the accuracy of classifiers using topics as features. 

The respository contains the scripts used in the project. Running the scripts requires the arXiv metadata dataset, which can be downloaded here: https://www.kaggle.com/datasets/Cornell-University/arxiv. The experiment connected to this project require multiple products from each script. Thus, the scripts need to be run multiple times in order to recreate the experiment. This project used the University of Groningen's Peregrine HPC to run all the scripts. However, any cloud computing cluster such AWS or Google Collab can be used. Note that, due to the time required to complete some of the scripts (BERTopic modeling in particular), using a paid instance may be expensive. 

What follows are instructions on how to get the results by recreating the experiment.

## 1. arXiv Preprocessing

The project uses the abstracts and categories from the arXiv dataset. It specifically uses the articles published after 2018. To get this subset, do:

- place the arXiv dataset in the "dataset" folder
- run "python prepare\_post2018\_arxiv.py"

You now have the dataset required for the following scripts.

## 2. Topic Modeling

The project uses two topic models for this experiments: Latent Dirichlet Allocation and BERTopic. The experiments use multiple versions of each topic model, varying in the topic size. Specifically, it uses topic sizes of 25, 50, 75, and 100. Furthemore, five models for each topic size were created. This results in a total of 40 topic models. 

Due to Peregrine's job batching system, each model version was run individually. Though it is possible to train all topic model versions in a single batch, this is only recommended if there is enough time and resources. Otherwise, the best option is to train each model individually. The scripts are run with the assumption that the preprocessed arXiv dataset is in the "dataset" folder and is being run in the "modeling\_scripts" folder.

Run the following line to get a specific model version:

- python model_\[model\_name\] \[t\] \[s\]

where "model\_name" refers to either "lda" or "bertopic", "t" refers to the topic size, and "s" refers to the model number. The models will be saved in the corresponding models folder (e.g. LDA models in "lda\_models"). It is recommended to train a total of five models for each topic size as the following coherence script assumes that amount.

## 3. Model Coherency

Four coherence measures are computed for each topic model. The measures are: UCI, UMass, NPMI, and Roder's CV. These measurements are made using Gensim's coherence model. For LDA, however, another package called "tmtoolkit" is used to compute the measures in parallel (and thus faster). The coherence scripts computes each measure for all topic models of a given topic size. For instance, it will compute the four measures for all five models with topic size 25. 

Run the following line to compute the coherence of a given set of topic models:

- python coherency_\[model\_name\] \\[t\]

where "model\_name" refers to either "lda" or "bertopic" and "t" refers to the topic size.
