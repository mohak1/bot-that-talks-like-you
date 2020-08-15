from model_classes import Encoder, Attention, Attention_Decoder
from model_classes import trainIters

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


CUDA = torch.cuda.is_available()
if CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

    

def start_training(voc, trimmed_pairs, save_dir, load_model_path=None):

    #training function
    teacher_forcing_ratio = 0.5

    # Configure models
    model_name = 'whatsapp_bot'
    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 5

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = load_model_path
    checkpoint_iter = 4000
    #loadFilename = os.path.join(save_dir, model_name, '{}_checkpoint.tar'.format(checkpoint_iter))


    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        print('Model loaded successfully')
    else:
        print('Building encoder and decoder ...')
    
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_of_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = Attention_Decoder(attn_model, embedding, hidden_size, hidden_size, voc.num_of_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')


    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.00000001
    decoder_learning_ratio = 5.0
    n_iteration = 500000
    print_every = 1000
    save_every = 1000

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, voc, trimmed_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, loadFilename, teacher_forcing_ratio, hidden_size)
