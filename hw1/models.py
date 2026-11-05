# models.py

from sentiment_data import *
from utils import *
import math
import random
import string

from collections import Counter

# Common English stopwords to filter out
# NOTE: Removed "no", "not", "n't", "nor" - CRITICAL for sentiment (negations)
STOPWORDS = {
    "a","an","and","are",  "as","at" , "be","been","but","by","for", "if", "in", "into", "is", "it", "of", "on",
    "or", "such","that","the","their" ,  "then",  "there","these","they", "this", "to", "was", "will", "with", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "when", "where", "why",  "have","has","had","do", "does", "did", "should", "would", "could",
    "about", "from", "up", "out", "all", "each", "same", "more", "most"}

class FeatureExtractor(object):
    """
        Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        counter = Counter() 
        processed_words = []
        # limpiar signos de puntuacion.
        for i in range(len( sentence) ):
            word = sentence[i].lower()
            #excetpo exclamacion e interrogacion,
            word =''.join(c if c in '!?' else (c if not c in string.punctuation else '') for c in word)

            if word:  
                processed_words.append(word)
        
        # extraer stopword
        for i in range(len(processed_words)):
            word = processed_words[i]
            # si word esta en Stopwords evitar
            if word in STOPWORDS: 
                continue

            # si tiene digitos
            if any(c.isdigit() for c in word): 
                continue 
            

            # normalizar negaciones
            negation_prefix = ""
            if i > 0 and processed_words[i-1] in {"not", "no", "n't", "nor"}:
                negation_prefix = "NOT_"
            feature_word = negation_prefix + word
                        
            feature_index = self.indexer.add_and_get_index(feature_word, add_to_indexer)
            if feature_index != -1:
                counter[feature_index] = 1  #marca la palabra en el contador
        return counter
            


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        counter = Counter() 
        processed_words = [] 

        #limpiar signos de puntuacion
        for i in range(len(sentence)):
            word = sentence[i].lower()
            #excepto exclamacion o interrogacion
            word = ''.join(c if c in '!?'  else (c  if not c in string.punctuation else '') for c in word)
            if word: #si la palabra no esta vacia atachala
                processed_words.append(word)
        
        #extraer bigramas con stopword
        for i in range(len(processed_words)-1):
            word1 = processed_words[i ] 
            word2 = processed_words[i +1] 
            # evadir bigramas con stopwords
            
            if word1 in STOPWORDS or word2 in STOPWORDS: 
                continue 
            
            # evadir bigramas con digitos
            if any(c.isdigit() for c in word1) or any(c.isdigit() for c  in word2):
                continue 
            
            # chequear por negacion
            negation_prefix = "" 
            if i > 0 and processed_words[i-1] in {"not", "no", "n't", "nor"}: 
                negation_prefix = "NOT_"
            
            #atachar negacion al primera palabra
            bigr  = (negation_prefix + word1,  word2) 
            feature_index = self.indexer.add_and_get_index(bigr, add_to_indexer)
            if feature_index != -1:
                counter[feature_index] = 1  # marcar al bigrama dentro del contador
        return counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) ->  Counter:
        counter = Counter() 
        processed_words =[] 
        #limpiar signos de puntuacion
        for i in range(len(sentence)): 
            word = sentence[i].lower() 
            # remover excepto excalamacion e interrogacion 
            word = ''.join(c if c in '!?' else  (c if not c in string.punctuation else '')  for c  in word) 
            #solo anade si no la palabra no esta vacia
            if word:  #
                processed_words.append(word)
        
        # Similar a bigramas extraeremos trigramas con negacion
        for i in range(len(processed_words) - 2):
            word1 = processed_words[i]
            word2 = processed_words[i + 1]
            word3 = processed_words[i + 2]
            # evadir trigramas con stopwords
            if word1 in STOPWORDS or word2 in STOPWORDS or word3 in STOPWORDS: 
                continue 
            #evadir trigrama con digito
            if any(c.isdigit() for c in word1) or any(c.isdigit() for c in  word2)  or any(c.isdigit()  for  c in word3): 
                continue 
            
            #chequear negacion o prefiijos
            negation_prefix ="" 
            if i > 0 and processed_words[i-1] in {"not", "no","n't","nor"}: 
                negation_prefix = "NOT_" 
            #anadir negacion a la primera palabra 
            trig = (negation_prefix + word1,word2,word3)

            feature_index = self.indexer.add_and_get_index(trig, add_to_indexer)

            
            if feature_index != -1:
                counter[feature_index] = 1  #anade el trigrama si no se encuenra
        return counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feature_extractor):
        self.w =weights 
        self.feature_extractor =feature_extractor

    def predict(self, sentence: List[str])-> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score =0.0 
        for index,value in features.items(): 
            if index <len(self.w): 
                score += self.w[index]*value
        
        pred = 0.0
        if score >0.0 :
            pred = 1.0
        return pred


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feature_extractor):
        self.w = weights
        self.feature_extractor = feature_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0
        for index , value in features.items():
            if index < len(self.w):
                score += self.w[index] * value
        # Limitar score para evitar overflow en exp()
        score =max(-300, min(300,score)) 
        sigmoid = 1 /(1+math.exp(- score) ) 
        if sigmoid > 0.5 :
            return 1
        else:
            return 0
        

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    num_features =  len(feat_extractor.get_indexer())
    w = [random.uniform(-0.01, 0.01) for _ in range(num_features)]
    num_epochs  = 50
    
    for epoch in range(num_epochs):
        for example in train_exs:
            features = feat_extractor.extract_features(example.words, add_to_indexer=True)
            score = sum(w[i] * features.get(i, 0) for i in features if i <len(w))
            prediction =0
            if score > 0.0  :
                prediction = 1
            
            if prediction !=  example.label:
                for i in features:
                    # Expandir weights si es necesario
                    while i >= len(w):
                        w.append(0.0)
                    w[i] += (example.label - prediction) * features[i]
    
    return PerceptronClassifier(w, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    lr = 0.15

    epochs  =  100
    decay_factor =0.95  #reducir weights by .
    decay_period =5
    num_features= len( feat_extractor.get_indexer())
    w = [random.uniform(-0.01, 0.01) for _ in range(num_features)]
    loss_historico = []  #trackin del loss
    
    for epoch in range(epochs):
        # Decay learning rate every decay_every epochs
        if epoch > 0 and epoch % decay_period == 0:
            lr *= decay_factor
        
        epoch_loss = 0.0 
        for example in train_exs: 
            features =feat_extractor.extract_features(example.words , add_to_indexer=True) 
            z = sum(w[i]* features.get(i,  0) for  i in features if i  <  len( w)) 
            z = min(z,300)  #estabilidad de calculo
            z = max(z,-300)  #estabilidad de calculo
            prediction = 1/(1 + math.exp( -z ) ) 
            
            # perdida Cross-entropia
            epsilon = 1e-6
            
            
            epoch_loss += -(example.label * math.log(max(prediction, epsilon)) + 
                           (1 -example.label) *math.log(max(1 - prediction, epsilon)))
            

            for i in features:
                # Expandir weights si es necesario
                
                while i >= len(w) :
                    w.append(0.0)
                w[i] += lr * (example.label - prediction) * features[i]
        
        loss_historico.append(epoch_loss / len(train_exs))
    
    # guardar la perdida
    classifier = LogisticRegressionClassifier(w, feat_extractor)
    classifier.loss_history = loss_historico
    return classifier



def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model