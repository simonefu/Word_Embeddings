import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class CBOW_model(nn.Module):
    """
    The class compute words embeddings using the CBOW architecture 

    ...

    Attributes
    ----------
    vocab_size : int
        maximum number of vocabulary size
    embedding_dim : int
        dimension of the word embeddings vector space
    context_size : int
        the range distance of the words that we need to predict the target word
    inputs : list
        is the list of indices associated with the words (id2words)
        

    Methods
    -------
    forward(self, inputs)
        describe the forward computation of the architecture 
    
    predict(self, input)
          
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
            """
            Parameters
            ----------
            vocab_size : int
                 maximum number of vocabulary size
            embedding_dim : int
                 dimension of the word embeddings vector space
            context_size : int
                 the range distance of the words that we need to predict the target word
            """
        super(CBOW_model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
       """Compute the probabilities for each word of the vocabulary

            Parameters
            ----------
            inputs : list
                   is the list of indices associated with the words (id2words)

            Raises
            ------
            NotImplementedError
                 If no words is init it raises error
       """
        embeds = self.embeddings(inputs).view((1, -1))  # -1 implies size inferred for that index from the size of the data
        out1 = F.relu(self.linear1(embeds)) # output of first layer
        out2 = self.linear2(out1)           # output of second layer
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs

    def predict(self,input):
           """get the word with the higher probability 

            Parameters
            ----------
            inputs : torch.tensor
                   the tensor containing the probability for each word of the vocabulary

       """
        context_idxs = torch.tensor([word_to_ix[w] for w in input], dtype = torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        #print(res_val)
        #print(res_ind)
        for arg in zip(res_val,res_ind):
            #print(arg)
            print([(key,val,arg[0]) for key,val in word_to_ix.items() if val == arg[1]])
            
class Skipgram_model(nn.Module):
    """
    The class compute words embeddings using the Skipgram architecture 

    ...

    Attributes
    ----------
    vocab_size : int
        maximum number of vocabulary size
    embedding_dim : int
        dimension of the word embeddings vector space
    context_size : int
        the range distance of the words that we need to predict the target word
    inputs : list
        is the list of indices associated with the words (id2words)
        

    Methods
    -------
    forward(self, inputs)
        describe the forward computation of the architecture 
    
    predict(self, input)
          
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
            """
            Parameters
            ----------
            vocab_size : int
                 maximum number of vocabulary size
            embedding_dim : int
                 dimension of the word embeddings vector space
            context_size : int
                 the range distance of the words that we need to predict the target word
            """
        super(Skipgram_model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, context_size * vocab_size)

    def forward(self, inputs):
       """Compute the probabilities for each word of the vocabulary

            Parameters
            ----------
            inputs : list
                   is the list of indices associated with the words (id2words)

            Raises
            ------
            NotImplementedError
                 If no words is init it raises error
       """
        embeds = self.embeddings(inputs).view((1, -1))  # -1 implies size inferred for that index from the size of the data
        out1 = F.relu(self.linear1(embeds)) # output of first layer
        out2 = self.linear2(out1)           # output of second layer
        log_probs = F.log_softmax(out2, dim=1).view(CONTEXT_SIZE,-1)
        return log_probs

    def predict(self,input):
       """get the word with the higher probability 

            Parameters
            ----------
            inputs : torch.tensor
                   the tensor containing the probability for each word of the vocabulary

       """
        context_idxs = torch.tensor([word_to_ix[input]], dtype=torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        indices = [res_ind[i][0] for i in np.arange(0,3)]
        for arg in indices:
            print( [ (key, val) for key,val in word_to_ix.items() if val == arg ])
           

def get_key(word_id):
    """get the word corresponfing to the id map

    Parameters
    ----------
    word_id : int
        the word id corresponding to the word of the vocabulary
    Returns
    -------
    word
        the corresponding word 
    """
    for key,val in word_to_ix.items():
        if(val == word_id):
            print(key)

def cluster_embeddings(filename,nclusters):
    """compute the most similar word  

    Parameters
    ----------
    filename : str
        the word id corresponding to the word of the vocabulary
    nclusters: int
        the number of clusters
    Returns
    -------
    word
        the corresponding word 
    """
    X = np.load(filename)
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X)
    center = kmeans.cluster_centers_
    distances = euclidean_distances(X,center)

    for i in np.arange(0,distances.shape[1]):
        word_id = np.argmin(distances[:,i])
        print(word_id)
        get_key(word_id)

def read_data(file_path):
    """ Read and process the text file by removing stopwords 

    Parameters
    ----------
    file_path : str
        the path where the input text file is stored
    Returns
    -------
    str
        the preprocessed text file 
    """
    tokenizer = RegexpTokenizer(r'\w+')
    data = urllib.request.urlopen(file_path)
    data = data.read().decode('utf8')
    tokenized_data = word_tokenize(data)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.',',',':',';','(',')','#','--','...','"'])
    cleaned_words = [ i for i in tokenized_data if i not in stop_words ]
    return(cleaned_words)
    
class CBOWModeler(nn.Module):
    """
    The class compute words embeddings using the CBOW architecture 

    ...

    Attributes
    ----------
    vocab_size : int
        maximum number of vocabulary size
    embedding_dim : int
        dimension of the word embeddings vector space
    context_size : int
        the range distance of the words that we need to predict the target word
    inputs : list
        is the list of indices associated with the words (id2words)
        

    Methods
    -------
    forward(self, inputs)
        describe the forward computation of the architecture 
    
    predict(self, input)
          
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        """
            Parameters
            ----------
            vocab_size : int
                 maximum number of vocabulary size
            embedding_dim : int
                 dimension of the word embeddings vector space
            context_size : int
                 the range distance of the words that we need to predict the target word
         """
        super(CBOWModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
       """Compute the probabilities for each word of the vocabulary

            Parameters
            ----------
            inputs : list
                   is the list of indices associated with the words (id2words)

            Raises
            ------
            NotImplementedError
                 If no words is init it raises error
       """
        #from IPython.core.debugger import Tracer; Tracer()()
        embeds = self.embeddings(inputs).view((1, -1))  # -1 implies size inferred for that index from the size of the data
        #print(np.mean(np.mean(self.linear2.weight.data.numpy())))
        out1 = F.relu(self.linear1(embeds)) # output of first layer
        out2 = self.linear2(out1)           # output of second layer
        #print(embeds)
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs

    def predict(self,input):
       """get the word with the higher probability 

            Parameters
            ----------
            inputs : torch.tensor
                   the tensor containing the probability for each word of the vocabulary

       """    
        context_idxs = torch.tensor([word_to_ix[w] for w in input], dtype=torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        #print(res_val)
        #print(res_ind)
        for arg in zip(res_val,res_ind):
            #print(arg)
            print([(key,val,arg[0]) for key,val in word_to_ix.items() if val == arg[1]])

    def freeze_layer(self,layer):
       """it does not upgrade the weights of the layer  

            Parameters
            ----------
            layer : str
                   the corresponding layer to freeze

       """    
        for name,child in model.named_children():
            print(name,child)
            if(name == layer):
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())
                    params.requires_grad= False

    def print_layer_parameters(self):
        for name,child in model.named_children():
                print(name,child)
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())

    def write_embedding_to_file(self,filename):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename,weights)
        return weights

