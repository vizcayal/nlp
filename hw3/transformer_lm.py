# models.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim



class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size
        
    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class Transformer_LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_tx_layers, dim_ff = 128, max_seq = 40):
        super().__init__()
        self.caract_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = num_heads, dim_feedforward= dim_ff, 
        batch_first=True)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers= num_tx_layers)
        self.linear_output = nn.Linear(d_model, vocab_size)
        self.max_seq = max_seq

    def forward(self, indices):
        device = indices.device
        seq_len = indices.shape[-1]
        position_ind = torch.arange(seq_len, device = device)
        casual_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device = device)
        emb = self.caract_emb(indices)
        emb_pos = emb + self.pos_emb(position_ind)
        attention = self.transformer(emb_pos, mask = casual_mask, is_causal = True)
        output = self.linear_output(attention)
        log_output = torch.log_softmax(output, dim = -1)
        return log_output




class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, seq_len = 40):
        self.model = model
        self.vocab_index = vocab_index
        self.seq_len = seq_len
        self.device = next(model.parameters()).device
        self.spc_ind = vocab_index.index_of(' ')

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            if len(context) == 0:
                indices = [self.spc_ind]
            else:
                truncated = context[-(self.seq_len - 1):]
                indices = [self.spc_ind] + [self.vocab_index.index_of(c) for c in truncated]
            indices_torch = torch.tensor([indices], dtype = torch.long, device = self.device)
            log_output = self.model(indices_torch)
            next_char_prob = log_output[0,-1,:].cpu().numpy()
            return next_char_prob


    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        total_log = 0.0
        curr_context = context
        

        for c in next_chars:
            prob_ind = self.get_next_char_log_probs(curr_context)
            ind_c = self.vocab_index.index_of(c)
            prob_c = prob_ind[ind_c]
            curr_context = curr_context + c
            total_log += prob_c
        return  float(total_log)

        


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ind = [vocab_index.index_of(c) for c in train_text]
    spc_ind = vocab_index.index_of(' ')
    seq_len = 40
    vocab_size = len(vocab_index)
    d_model = 64
    num_tx_layer = 2
    heads = 4
    model = Transformer_LM( vocab_size, d_model, num_heads = heads, num_tx_layers=num_tx_layer, dim_ff = 128, max_seq = 40).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    input_chunks = []
    target_chunks = []
    loss_funct = nn.NLLLoss()

    #prepare inputs and outputs
    for i in range(0, len(train_ind) - seq_len, seq_len):
        chunk = train_ind[i: i+seq_len]
        inp = [spc_ind] + chunk[:-1]
        target = chunk
        input_chunks.append(inp)
        target_chunks.append(target)

    input_tensors = torch.tensor(input_chunks, dtype = torch.long)
    target_tensors = torch.tensor(target_chunks, dtype = torch.long)


    batch_size = 32
    dataset = TensorDataset(input_tensors, target_tensors)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    num_epoch = 10

    for ep in range(num_epoch):
        model.train()
        total_loss = 0.0
        
        for batch_inputs, batch_target in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_target = batch_target.to(device)
            probs = model(batch_inputs)
            loss = loss_funct(probs.view(-1, len(vocab_index)), batch_target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {ep + 1} - Loss: {total_loss / len(train_loader):.4f}")
    
    return NeuralLanguageModel(model, vocab_index)
