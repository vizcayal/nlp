# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import copy


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers=1):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.positions = num_positions
        self.emb_dim = d_model
        self.kq_dim = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_dim)
        self.positional_emb = PositionalEncoding(self.emb_dim, self.positions)
        self.transformer_layers = nn.ModuleList([TransformerLayer(self.emb_dim, self.kq_dim) for _ in range(self.num_layers)])
        self.ff_layer = nn.Linear(self.emb_dim, self.num_classes)
        
        

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        embds=self.embedding_layer(indices)
        embs_pos= self.positional_emb(embds)
        attention_maps = []
        for layer in self.transformer_layers:
            embs_pos, attention = layer(embs_pos)
            attention_maps.append(attention)
        output = self.ff_layer(embs_pos)
        return torch.log_softmax(output, dim=-1), attention_maps




# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.embed_dim = d_model
        self.qv_dim = d_internal
        self.w_k = nn.Linear(self.embed_dim, self.qv_dim)
        self.w_q = nn.Linear(self.embed_dim, self.qv_dim)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.ff1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.ff2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.non_linear = nn.ReLU()

    def forward(self, input_vecs):
        # calculate K 
        K=self.w_k(input_vecs) #(N, d_internal)
        # calculate Q 
        Q =self.w_q( input_vecs) #(N, d_internal)
        # calcualte V 
        V=self.w_v( input_vecs ) #(N, d_model)
        #calculate scores 
        scores = Q @ K.transpose(-2,-1) #(N, N)
        # normalize by qv_dim 
        scores = scores/(self.qv_dim**0.5) #(N, N)
        # calculate attention
        attention = torch.softmax(scores, dim=-1) #(N, N)
        # multiply by V 
        self_attention = attention @ V #(N, d_model)
        #residual connection 1
        input_vecs= input_vecs + self_attention 

        #linear proyection 1
        ff1 = self.ff1(input_vecs)
        #non-linearity
        non_linear = self.non_linear(ff1)
        #linear proyection 2
        ff2 = self.ff2(non_linear)
        #residual connection 2
        output = input_vecs + ff2
        return output, attention



# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    
    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    
    
    vocab_size = 27       # 26 letras minúsculas ('a'-'z') + espacio ' '
    num_positions = 20    # La longitud de cada secuencia es exactamente 20 caracteres
    d_model = 64          # Dimensión de embedding y capas (ej. 32 o 64)
    d_internal = 32       # Dimensión interna para K y Q (ej. 32)
    num_classes = 3       # Clases de salida: 0, 1 o 2 (veces que apareció la letra antes)
    num_layers = 1        # Número de capas del Transformer (1 o 2 capas son suficientes)


    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()
        
    num_epochs = 10
    best_acc = 0.0
    best_weights = None
        
    for t in range(0, num_epochs):
        model.train()
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            logs,_ = model.forward(ex.input_tensor)
            loss = loss_fcn(logs, ex.output_tensor) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for ex in dev:
                logs, _ =model.forward(ex.input_tensor)
                preds = logs.argmax(dim = -1)
                correct += (preds == ex.output_tensor).sum().item()
                total += len(ex.output_tensor)
            dev_acc = correct / total
        print(f"epoch {t +1} | Loss Train: {loss_this_epoch:.2f} | dev_acc {dev_acc:.4f}")
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_weights = copy.deepcopy(model.state_dict())
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
