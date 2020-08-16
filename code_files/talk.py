from model_classes import GreedySearchDecoder
from model_classes import evaluate
from model_classes import Encoder, Attention_Decoder
from vocab import Vocabulary
from train import start_training
from preprocessing import PreprocessChats
import torch
import torch.nn as nn
import os 

#load the model
LOAD_MODEL_PATH = 'path_to_repository/whatsapp_bot/'

#LOAD_MODEL_PATH = '/home/mohak/Music/chatbot/whatsapp_bot/'
#find the most recent model checkpoint
l = os.listdir(LOAD_MODEL_PATH)
recent = 0
for i in l:
    if '_checkpoint.tar' not in i:
        continue
    i = i.strip('_checkpoint.tar')
    if int(i)>int(recent):
        recent = i

LOAD_MODEL_PATH = LOAD_MODEL_PATH + recent + '_checkpoint.tar'

#checkpoint = torch.load(LOAD_MODEL_PATH)
checkpoint = torch.load(LOAD_MODEL_PATH, map_location=torch.device('cpu'))

encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Vocabulary()
voc.__dict__ = checkpoint['voc_dict']

#do not edit these parameters
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

embedding = nn.Embedding(voc.num_of_words, hidden_size)
embedding.load_state_dict(embedding_sd)
encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)   
encoder.load_state_dict(encoder_sd)
decoder = Attention_Decoder(attn_model, embedding, hidden_size, hidden_size, voc.num_of_words, decoder_n_layers, dropout)
decoder.load_state_dict(decoder_sd)
print('Models loaded.')


def evaluate_input(encoder, decoder, searcher, voc):
    input_sentence = ''
    chat = PreprocessChats('', '', '', '')
    print('Start typing and press enter when you\'re done.')
    print('Type \'quit\' or \'q\' to stop chatting with the bot.')
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = chat.normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'END' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
            print('\n')

        except KeyError:
            print("Error: Encountered unknown word.")


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, voc)

# Begin chatting (uncomment and run the following line to begin)
evaluate_input(encoder, decoder, searcher, voc)
