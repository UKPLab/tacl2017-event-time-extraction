# Event Time Extraction with a Decision Tree of Neural Classifiers
The following repository contains the Python code for a decision tree that applies neural network classifiers at its nodes. It was used for our publication [https://transacl.org/ojs/index.php/tacl/article/download/1218/282](Event Time Extraction with a Decision Tree of Neural Classifiers).


# Citation
If you find the implementation useful, please cite the following paper: Event Time Extraction with a Decision Tree of Neural Classifiers (to appear)

```
@article{TACL1218,
        author = {Reimers, Nils  and Dehghani, Nazanin  and Gurevych, Iryna },
        title = {Event Time Extraction with a Decision Tree of Neural Classifiers},
        journal = {Transactions of the Association for Computational Linguistics},
        volume = {6},
        year = {2018},
        issn = {2307-387X},
        url = {https://transacl.org/ojs/index.php/tacl/article/view/1218},
        pages = {77--89}
}
``` 

> **Abstract:** Extracting the information from text when an event happened is challenging. Documents do not only report on current events, but also on past events as well as on future events. Often, the relevant time information for an event is scattered across the document. In this paper we present a novel method to automatically anchor events in time. To our knowledge it is the first approach that takes temporal information from the complete document into account. We created a decision tree that applies neural network based classifiers at its nodes. It infers stepwise the final information at which date or at which time frame an event happened. We evaluate the approach on the TimeBank-EventTime Corpus achieving an accuracy of 42.0% compared to an inter-annotator agreement (IAA) of 56.7%. For events that span over a single day we observe an accuracy improvement of 33.4 points compared to the state-of-the-art CAEVO system. Without re-training, we apply this model to the SemEval-2015 Task 4 on automatic timeline generation and achieve an improvement of 4.01 points F_1-score compared to the state-of-the-art.


Contact person: Nils Reimers, reimers@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


# Requirements

This code was developed and tested with:
- Python 2.7 on Ubuntu 16.04
- Theano 0.9.0
- Keras 0.3.3
- NLTK 3.2.5

It does not run with more recent versions of Keras or with Python 3. The Theano backend must be used for Keras.

The python package requirements can be found in the `requirements.txt` folder.

Futher, it requires requires the dependency based word embeddings by Levy et al.: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

To run the code, perform the following steps:
* Run `1_CreatePKLFiles.py`. This loads the embeddings file and generates pickle files (stored in the pkl-folder) that are later used for training the network.
* The `2_Train_*.py` scripts trains networks for the individual nodes in the decision tree. Run `2_TrainAllModels.py` to train all models. The code performs a random hyperparameter search, samples parameters at random, trains the network and stores the output.
* The `3_CreateOutput.py` and `4_CreateLabels.py` plugs all nodes together and generates the final event time output.

 

