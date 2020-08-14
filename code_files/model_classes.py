import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vocab import Vocabulary
import random
import os 

CUDA = torch.cuda.is_available()
if CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

MAX_LENGTH = 20

def input_var(index_matrix, voc):
    indexes_batch = [voc.index_from_sentence(sentence) for sentence in index_matrix]
    lengths = torch.Tensor([len(indexes) for indexes in indexes_batch])
    padlist = voc.zero_padding(indexes_batch)
    pad_var = torch.LongTensor(padlist)
    return pad_var, lengths

def output_var(index_matrix, voc):
    indexes_batch = [voc.index_from_sentence(sentence) for sentence in index_matrix]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padlist = voc.zero_padding(indexes_batch)
    mask = voc.binary_matrix(padlist)
    mask = torch.ByteTensor(mask)
    padvar = torch.LongTensor(padlist)
    return padvar, mask, max_target_len

#return all items for a given batch of pairs
def train_data_batch(voc, pair_batch):
    #sort the questions in decreasing length
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch = []
    output_batch = []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size=hidden_size , hidden_size=hidden_size, num_layers = n_layers, dropout=dropout, bidirectional = True)

    def forward(self, input_seq, input_length, hidden=None):
        #input_seq: input sentences of dim (max_len, batch_size)
        #input_length: list containing len of each sentence in the batch
        #hidden_state of dim: (n_layers * num of directions, batch_size, hidden_size)
    
        #converting word sequences to embeddings
        embedded = self.embedding(input_seq)
        #pack padded batch of sequence for RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        #GRU layer
        gru_outputs, hidden = self.gru(packed, hidden)
        #unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_outputs)
        #adding the bidirectoinal GRU outputs
        outputs_sum = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        #outputs_sum dim: (max_length, batch_size, hidden_size)
        #hidden: hidden state for last timestamp, dim: (n_layers)
        return outputs_sum, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
  
    def forward(self, hidden, encoder_output):
        atten_energies = self.dot_score(hidden, encoder_output)
        atten_energies = atten_energies.t()
        return F.softmax(atten_energies, dim=1).unsqueeze(1)


class Attention_Decoder(nn.Module):
    def __init__(self, atten_model, embedding, input_size, hidden_size, output_size, n_layers=1, dropout=0.01):
        super(Attention_Decoder, self).__init__()
        self.atten_model = atten_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        #layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.atten = Attention(atten_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_output):
        #input_step: one time step (one word) of the input sentence batch. dim: (1,batch_size)
        #last_hidden: final hidden state of encoder GRU. dim: (n_layers*num_direction, batch_size, hidden_size)
        #encoder_output: output of encoder model. dim: (seq_len, batch, num_directions*hidden_size)
        #one batch of words are processed in one forward pass

        #get embedding of current word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        #Forward pass through GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #rnn_output: (1, batch_size, num_directions*hidden_size)
        #hidden: (num_layers*num_directions, batch, hidden_size)
        atten_weights = self.atten(rnn_output, encoder_output)
        context = atten_weights.bmm(encoder_output.transpose(0, 1))
        #bmm is batch matrix-multiplication-function
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        #predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden
        #output: output of the softmax probability list
        #hidden: final hidden state of the GRU. dim: (n_layers*num_directions, batch_size, hidden_size)

#the input is padded but the padded words should not be used in loss calculation
def mask_NLL_loss(decoder_out, target, mask):
    #NLL: negative log likelihood
    #mask is a binary matrix which represents the presence of word
    #get total words in the sentence
    n_total = mask.sum()
    target = target.view(-1,1)
    gather_tensor = torch.gather(decoder_out, 1, target)
    #negative log likelyhood loss 
    cross_entropy = -torch.log(gather_tensor)
    #select the non zero elements
    loss = cross_entropy.masked_select(mask)
    loss = loss.mean()
    loss = loss.to(device)
    return loss, n_total.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optim, decoder_optim,
          batch_size, clip, voc, teacher_forcing_ratio, max_length=MAX_LENGTH):

    
    #zero gradients
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    #send to gpu
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    #initialize loss variables
    loss=0
    print_losses = []
    n_totals=0

    #forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    #create initial decoder input, start with START token
    decoder_input = torch.LongTensor([[voc.START_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)    
  
    #set initial decoder hidden state to the encoders final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    #determine the amount of teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    #forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            #teacher forcing the next input
            decoder_input = target_variable[t].view(1,-1)
      
            #calculate loss
            mask_loss, n_total = mask_NLL_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            #no teacher forcing. next input is decoder output 
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range (batch_size)]])
            decoder_input = decoder_input.to(device)
            #calculate loss
            mask_loss, n_total = mask_NLL_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    #perform backprop
    loss.backward()

    #gradient clipping: handling vanishing and exploding gradient problem
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    #adjust model weights
    encoder_optim.step()
    decoder_optim.step()
  
    return sum(print_losses)/n_totals


def trainIters(model_name, voc, trimmed_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, loadFilename, teacher_forcing_ratio, hidden_sizef):

    # Load batches for each iteration
    training_batches = [train_data_batch(voc, [random.choice(trimmed_pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = int(loadFilename.strip('_checkpoint.tar').split('/')[-1]) + 1
        
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, voc, teacher_forcing_ratio)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, voc):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device='cpu', dtype=torch.long) * self.voc.START_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device='cpu', dtype=torch.long)
        all_scores = torch.zeros([0], device='cpu')
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [voc.index_from_sentence(sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to('cpu')
    lengths = lengths.to('cpu')
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

