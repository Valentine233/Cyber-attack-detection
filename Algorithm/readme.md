This project is for detecting cyber attacks.

# CODES

## files directly under folder codes
### config.py
This file is for defining file directories and values of hyper parameters. Each person can set their own configrations.

### utils.py
Some general and useful functions are written here.

- **to_var**
Transform a numpy to a variable stored in a given gpu.

- **getSent**
Get sentence and label pairs from data files and save them into files.
Here, we only extract requests as sentences.


## folder "data_preprocessing"
### data_preprocessing.py
This file is for transforming string sentences to features that can be fed to networks.

- **One_hot**
Transform sentences to one hot features for language model.
Each char is represented by a one hot vector whose length is number of total chars.

- **GetFeatures**
Get all sentences represented by one hot features at one time. Note that this class is intended for getting intermediate features, so the order of data can't be changed.

- **Bigram**
The bigram version of One_hot, with which it shares the same APIs.


## folder "networks"
### autoencoder.py
We use autoencoder to extract features. The output of encoder is the final feature we want.

- **SimpleAutoEncoder**
A simple autoencoder containing only linear and activation layers

- **VAE**
A Variational Autoencoder

### rnnlm.py
A RNN language model which is trained to predict next word given the current word. After one sentence has been fed into the net, the hidden state of RNN is the final feature of this sentence.

- **RNNLM**
Network for RNN language model


## folder "features"
### train_rnnml.py
train an RNN language model

### train_ae.py
train an autoencoder model with a trained RNN language model

### get_rnnlm_features.py
save intermediate features of RNN language model into a file given a trained RNN language model

### get_ae_features.py
save intermediate features of autoencoder model into a file given a trained autoencoder model


## folder "classification"
### train_mlp.py
After extracting features of sentences, do a classification with multi-layer perceptron. Evaluation metrics like accuracy, precision and recall are calculated. 2d-PCA is performed in the end for visualization.


## folder "clustering"
### train_kmeans.py
After extracting features of sentences, do a clustering with kmeans. Evaluation metrics like accuracy, precision and recall are calculated. 2d-PCA is performed in the end for visualization.



# DATA

store dataset and intermediate features
## HTTP_2010
[HTTP dataset CSIC 2010](http://www.isi.csic.es/dataset/)


## emb128-hid64
intermediate features after rnnlm and ae with embed_size=128 and hidden_size=64


## emb200-hid128
intermediate features after rnnlm and ae with embed_size=200 and hidden_size=128



# MODEL

## emb128-hid64
models and logs of rnnlm and ae with embed_size=128 and hidden_size=64


## emb200-hid128
models and logs rnnlm and ae with embed_size=200 and hidden_size=128



