# models.py

from sentiment_data import read_word_embeddings
from torch import dropout
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class PrefixEmbeddings:

    def __init__(self, prefix_ind, vectors):
        self.word_indexer = prefix_ind
        self.vectors= vectors

    def get_initialized_embedding_layer(self, frozen = False):
        vectors = torch.FloatTensor(self.vectors)
        embedding = nn.Embedding.from_pretrained(vectors, freeze = frozen)
        return embedding
    
    def get_embedding_length(self):
        return len(self.vectors[0])
    
    def get_prefix(self, word:str) -> str:
        prefix = word[:3]
        return prefix

    def get_index(self, word:str)-> int:
        pref = self.get_prefix(word)
        ind = self.word_indexer.index_of(pref)
        if ind != -1:
            return ind
        return self.word_indexer.index_of("UNK")

def build_prefix_embeddings(word_embeddings: WordEmbeddings, train_exs: List[SentimentExample] = None) -> PrefixEmbeddings:
    from collections import defaultdict
    prefix_to_vectors = defaultdict(list)

    for i in range(len(word_embeddings.word_indexer)):
        word = word_embeddings.word_indexer.get_object(i)
        if word not in ["PAD", "UNK"] and word is not None:
            prefix = word[:3]
            prefix_to_vectors[prefix].append(word_embeddings.vectors[i])
    
    prefix_indexer = Indexer()
    prefix_indexer.add_and_get_index("PAD")
    prefix_indexer.add_and_get_index("UNK")
    dim = word_embeddings.get_embedding_length()
    vectors = [np.zeros(dim), np.zeros(dim)]
    
    for prefix, vecs in prefix_to_vectors.items():
        prefix_indexer.add_and_get_index(prefix)
        avg_vec  = np.mean(vecs, axis = 0)
        vectors.append(avg_vec)
    
    if train_exs is not None:
        for ex in train_exs:
            for w in ex.words:
                p = w[:3]
                if not prefix_indexer.contains(p):
                    prefix_indexer.add_and_get_index(p)
                    vectors.append(np.random.normal(scale=0.01, size=dim))
        
    
    vectors = np.array(vectors)
    return PrefixEmbeddings(prefix_indexer,vectors)


class DANModule(nn.Module):

    def __init__(self, word_emb:WordEmbeddings, hidden_dim: int = 20, dropout = 0.2, num_classes:int = 2, frozen:bool = True): 

        super().__init__() 
        self.word_emb = word_emb.get_initialized_embedding_layer(frozen=frozen)
        self.hidden_dim = hidden_dim 
        self.clases = num_classes
        self.embed_dim = word_emb.get_embedding_length()
        self.layer1=nn.Linear(self.embed_dim,hidden_dim)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=dropout)
        self.layer_out = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1) 
        
        self.indexer = word_emb.word_indexer

    def forward(self, input_indices: torch.Tensor):

        if input_indices.dim() == 1:
            input_indices = input_indices.unsqueeze(0)
        
        embed = self.word_emb(input_indices)
        av = torch.mean(embed, dim = 1 )
        hidden = self.relu(self.layer1(av))
        hidden = self.dropout(hidden)
        out= self.layer_out(hidden )
        return self.log_softmax(out ) 
        
class PrefixSentimentClassifier(SentimentClassifier):
    def __init__(self, model: DANModule, prefix_embeddings: PrefixEmbeddings):
        self.model = model
        self.prefix_embeddings = prefix_embeddings

    def predict(self, x_words: List[str], has_typos: bool) -> int:
        self.model.eval()
        indices = [self.prefix_embeddings.get_index(w) for w in x_words]
        if len(indices) == 0:
            indices = [self.prefix_embeddings.get_index("UNK")]
        tensor = torch.tensor(indices, dtype = torch.long)
        with torch.no_grad():
            log_probs = self.model(tensor)
        return torch.argmax(log_probs, dim=-1).item()

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model : DANModule, word_embeddings: WordEmbeddings) :
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, x_words: List[str] , has_typos: bool) -> int:
        self.model.eval()
        
        unk_idx = self.model.indexer.index_of("UNK")
        indices = [
            self.model.indexer.index_of(w) if self.model.indexer.index_of(w) != -1 else unk_idx 
            for w in x_words
        ]
        
        if len(indices) == 0:
            indices = [unk_idx]

        tensor = torch.tensor(indices, dtype=torch.long)
        with torch.no_grad():
            log_probs = self.model(tensor) 

        return torch.argmax(log_probs, dim=-1).item()





def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    if train_model_for_typo_setting:
        emb = build_prefix_embeddings(word_embeddings, train_exs)
        frozen = False
        epochs = min(args.num_epochs, 2)
    else:
        emb= word_embeddings
        frozen = True
        epochs = args.num_epochs 
    
    
    model = DANModule(emb, args.hidden_size, frozen = frozen)

    loss_function = nn.NLLLoss()
    optmr = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    batchsize=args.batch_size
    indexer= emb.word_indexer
    unk_idx = indexer.index_of("UNK")

    train_data = []
    for example in train_exs:
        indices =  [indexer.index_of(w) if indexer.index_of(w) != -1 else unk_idx for w in example.words]

        if train_model_for_typo_setting:
            indices = [emb.get_index(w) for w in example.words]
        else:
            indices = [indexer.index_of(w) if indexer.index_of(w)!= -1 else unk_idx for w in example.words]
        
        if len(indices) == 0:
            indices = [unk_idx]
        
        x = torch.tensor(indices, dtype = torch.long)
        label = torch.tensor([example.label], dtype = torch.long)
        train_data.append([x,label])


    for ep in range(epochs):
        model.train()
        random.shuffle(train_data)
        t_loss = 0.00
        for x,label in train_data:
            optmr.zero_grad()
            probs = model(x)
            loss = loss_function(probs, label)
            loss.backward()
            optmr.step()
            t_loss = t_loss +loss.item()
        
        print(f"epoch {ep +1}/{epochs} - Loss: {t_loss/len(train_data):.3f}")

    if train_model_for_typo_setting:
        return PrefixSentimentClassifier(model, emb)
    else:
        return NeuralSentimentClassifier(model, emb)




        

        
            
