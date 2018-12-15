# from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import logging

# one hot sentence features for language model
class One_hot:
    def __init__(self, train_sents, test_sents, train_labels, test_labels, batch_size, shuffle=True, mode='char'):
        self.x_train, self.x_test, self.y_train, self.y_test = train_sents, test_sents, train_labels, test_labels
        self.batch_size = batch_size
        self.mode = mode
        # self.sentences_no_repeat = list(set(self.sentences))
        self.unigram2index = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.index2unigram = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}
        self.char_count = {}
        self.build()
        self.makeFeatures()

    def build(self, seuil=10, truncating=True):
        self.sentences = self.x_train + self.x_test
        if self.mode == 'char':
            for sentence in self.sentences:
                for char in sentence:
                    if char not in self.unigram2index:
                        self.char_count[char] = 1
                    else:
                        self.char_count[char] += 1
        elif self.mode == 'bpe':
            for sentence in self.sentences:
                words = sentence.split()
                for word in words:
                    if word not in self.unigram2index:
                        self.char_count[word] = 1
                    else:
                        self.char_count[word] += 1
        else:
            raise ValueError("Not support mode %s" % self.mode)
        if truncating == True:
            for char in self.char_count:
                if self.char_count[char] > seuil:
                    self.unigram2index[char] = len(self.unigram2index)
        self.index2unigram = {v: k for k, v in self.unigram2index.items()}

    def makeFeatures(self):
        self.train_features = []
        for i, sentence in enumerate(self.x_train):
            train_feature = []
            for char in sentence:
                if char not in self.unigram2index:
                    train_feature.append(self.unigram2index['<unk>'])
                else:
                    train_feature.append(self.unigram2index[char])
            # feature = torch.FloatTensor(train_feature)
            self.train_features.append(train_feature)

        self.test_features = []
        for i, sentence in enumerate(self.x_test):
            test_feature = []
            for char in sentence:
                if char not in self.unigram2index:
                    test_feature.append(self.unigram2index['<unk>'])
                else:
                    test_feature.append(self.unigram2index[char])
            # feature = torch.FloatTensor(test_feature)
            self.test_features.append(test_feature)

        self.all_features = self.train_features + self.test_features
        self.all_y = self.y_train + self.y_test

    def train_len(self):
        return len(self.x_train)

    def test_len(self):
        return len(self.x_test)

    def all_len(self):
        return len(self.x_train) + len(self.x_test)

    def voc_len(self):
        return len(self.unigram2index)

    def getBatch(self, idx, train=True):
        # shuffle before each epoch
        self.tr_features, self.y_tr = self.train_features, self.y_train
        if idx==0 and train:
            self.tr_features, self.y_tr = shuffle(self.train_features, self.y_train)
        
        if train:
            sent = self.tr_features
            label = self.y_tr
        else:
            sent = self.test_features
            label = self.y_test
        
        batch_sentences = []
        batch_labels = []
        batch_lens = []
        for i in range(self.batch_size):
            batch_sentences.append(sent[idx * self.batch_size + i])
            batch_labels.append(int(label[idx * self.batch_size + i]))
            batch_lens.append(len(sent[idx * self.batch_size + i]))

        return batch_sentences, batch_labels, batch_lens

    def getAllBatch(self, idx, train=True):
        # shuffle before each epoch

        self.features, self.y = self.all_features, self.all_y
        if idx==0 and train:
            self.features, self.y = shuffle(self.all_features, self.all_y)

        sent = self.features
        label = self.y
        
        batch_sentences = []
        batch_labels = []
        batch_lens = []
        for i in range(self.batch_size):
            batch_sentences.append(sent[idx * self.batch_size + i])
            batch_labels.append(int(label[idx * self.batch_size + i]))
            batch_lens.append(len(sent[idx * self.batch_size + i]))

        return batch_sentences, batch_labels, batch_lens


# # get all features
# class GetFeatures:
#     def __init__(self, train_sents, test_sents):
#         self.train_sents = train_sents
#         self.test_sents = test_sents
#         # self.sentences_no_repeat = list(set(self.sentences))
#         self.unigram2index = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
#         self.index2unigram = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}
#         self.build()
#         self.makeFeatures()

#     def build(self, seuil=10, truncating=True):
#         for sentence in self.train_sents:
#             for char in sentence:
#                 if char not in self.unigram2index:
#                     self.unigram2index[char] = len(self.unigram2index)
#         self.index2unigram = {v: k for k, v in self.unigram2index.items()}

#     def makeFeatures(self):
#         self.features = []
#         for i, sentence in enumerate(self.test_sents):
#             feature = []
#             for char in sentence:
#                 if char not in self.unigram2index:
#                     feature.append(self.unigram2index['<unk>'])
#                 else:
#                     feature.append(self.unigram2index[char])
#             self.features.append(feature)

#     def voc_len(self):
#         return len(self.unigram2index)

# make the bigram feature
class Bigram:
    def __init__(self, sentences, labels, batch_size, shuffle=True):
        self.sentences = sentences
        self.batch_size = batch_size
        # self.sentences_no_repeat = list(set(self.sentences))
        self.bigram2index = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.index2bigram = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}
        self.bigram_count = {}
        self.build()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(sentences, labels, test_size=0.2, shuffle=shuffle, random_state=1)
        self.makeFeatures()
        
    def build(self,seuil=2,truncating=True):
        for sentence in self.sentences:
            for char in sentence:
                if char not in self.bigram2index:
                    self.bigram2index[char] = len(self.bigram2index)
            bisent = [[sentence[i],sentence[i+1]] for i in range(len(sentence)-1)]   
            for term in bisent:
                bichar = term[0] + term[1]
                if bichar in self.bigram_count:
                    self.bigram_count[bichar] += 1
                else:
                    self.bigram_count[bichar] = 1
        if truncating == True:
            truncating_count = 0
            for bichar in self.bigram_count:
                if self.bigram_count[bichar] >= seuil:
                    self.bigram2index[bichar] = len(self.bigram2index)
                else:
                    truncating_count += 1
            print("Number of bichar truncated:",truncating_count)
        self.index2bigram = {v: k for k, v in self.bigram2index.items()}
    
    def makeFeatures(self):
        self.train_features = []
        for i, sentence in enumerate(self.x_train):
            train_feature = []
            for char in sentence:
                if char not in self.bigram2index:
                    train_feature.append(self.bigram2index['<unk>'])
                else:
                    train_feature.append(self.bigram2index[char])
            bisent = [[sentence[i],sentence[i+1]] for i in range(len(sentence)-1)]
            for term in bisent:
                bichar = term[0] + term[1]
                if bichar not in self.bigram2index:
                    train_feature.append(self.bigram2index['<unk>'])
                else:
                    train_feature.append(self.bigram2index[bichar])
            # feature = torch.FloatTensor(train_feature)
            self.train_features.append(train_feature)

        self.test_features = []
        for i, sentence in enumerate(self.x_test):
            test_feature = []
            for char in sentence:
                if char not in self.bigram2index:
                    test_feature.append(self.bigram2index['<unk>'])
                else:
                    test_feature.append(self.bigram2index[char])
            bisent = [[sentence[i],sentence[i+1]] for i in range(len(sentence)-1)]
            for term in bisent:
                bichar = term[0] + term[1]
                if bichar not in self.bigram2index:
                    test_feature.append(self.bigram2index['<unk>'])
                else:
                    test_feature.append(self.bigram2index[bichar])
            # feature = torch.FloatTensor(train_feature)
            self.test_features.append(test_feature)

    def train_len(self):
        return len(self.x_train)

    def test_len(self):
        return len(self.x_test)

    def voc_len(self):
        return len(self.unigram2index)

    def getBatch(self, idx, train=True):
        # shuffle before each epoch
        if idx==0 and train:
            self.train_features, self.y_train = shuffle(self.train_features, self.y_train)
        
        if train:
            sent = self.train_features
            label = self.y_train
        else:
            sent = self.test_features
            label = self.y_test
        
        batch_sentences = []
        batch_labels = []
        batch_lens = []
        for i in range(self.batch_size):
            batch_sentences.append(sent[idx * self.batch_size + i])
            batch_labels.append(int(label[idx * self.batch_size + i]))
            batch_lens.append(len(sent[idx * self.batch_size + i]))

        return batch_sentences, batch_labels, batch_lens 
    
UNK_SYMBOL = "<unk>"
UNK_INDEX = 0
UNK_VALUE = lambda dim: np.zeros(dim) # get an UNK of a specificed dimension

class Glove:
    """
    Stores pretrained word embeddings for GloVe, and
    outputs a Keras Embeddings layer.
    """
    def __init__(self, fn, dim = None):
        """
        Load a GloVe pretrained embeddings model.
        fn - Filename from which to load the embeddings
        dim - Dimension of expected word embeddings, used as verficiation,
              None avoids this check.
        """
        self.fn = fn
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self._load(self.fn)
        logging.debug("Done!")

    def _load(self, fn):
        """
        Load glove embedding from a given filename
        """
        self.word_index = {UNK_SYMBOL : UNK_INDEX}
        emb = []
        for line in open(fn):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if self.dim:
                assert(len(coefs) == self.dim)
            else:
                self.dim = len(coefs)

            # Record mapping from word to index
            self.word_index[word] = len(emb) + 1
            emb.append(coefs)

        # Add UNK at the first index in the table
        self.emb = np.array([UNK_VALUE(self.dim)] + emb)
        # Set the vobabulary size
        self.vocab_size = len(self.emb)

    def get_word_index(self, word, lower = False):
        """
        Get the index of a given word (int).
        If word doesnt exists, returns UNK.
        lower - controls whether the word should be lowered before checking map
        """
        if lower:
            word = word.lower()
        return self.word_index[word] \
            if (word in self.word_index) else UNK_INDEX

    def get_embedding_matrix(self):
        return self.emb