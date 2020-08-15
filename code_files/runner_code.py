from preprocessing import PreprocessChats
import csv
import codecs
from vocab import Vocabulary
import model_classes
from train import start_training
import os

#Path to be changed as suggested
CHAT_FOLDER = 'path_to_repository/whatsapp_chats/'
MASTER_FOLDER = 'path_to_repository/processed_chats/'
SAVE_MODEL = 'path_to_repository/'
YOUR_NAME = 'Your name as it appears in whatsapp settings'
LOAD_MODEL_PATH = SAVE_MODEL + 'whatsapp_bot/'

# CHAT_FOLDER = '/home/mohak/Music/chatbot/whatsapp_chats/'
# MASTER_FOLDER = '/home/mohak/Music/chatbot/processed_chats/'
# SAVE_MODEL = '/home/mohak/Music/chatbot/'
# YOUR_NAME = 'Mohak'

#delete the 'to be removed.txt' files
try:
    file = 'to be removed.txt'
    os.remove(MASTER_FOLDER+file)
    os.remove(CHAT_FOLDER+file)
    os.remove(LOAD_MODEL_PATH+file)
except:
    pass

#find the most recent model checkpoint
l = os.listdir(LOAD_MODEL_PATH)
recent = 0
for i in l:
    if '_checkpoint.tar' not in i:
        continue
    i = i.strip('_checkpoint.tar')
    if int(i)>int(recent):
        recent = i
if not os.path.exists((LOAD_MODEL_PATH + str(recent) + '_checkpoint.tar')):
    load_model_path = None
else:
    LOAD_MODEL_PATH = LOAD_MODEL_PATH + str(recent) + '_checkpoint.tar'
    load_model_path = LOAD_MODEL_PATH

MAX_LENGTH = 20

process = PreprocessChats(CHAT_FOLDER, MASTER_FOLDER, YOUR_NAME, MAX_LENGTH)

if len(os.listdir(MASTER_FOLDER))<2:
    #the folder is empty, chats have not been processed
    process.chat_rename()
    process.remove_auto_messages()

pairs = process.create_pairs()

#saving the formatted pairs
path_to_savefile = MASTER_FOLDER+'all_chat_pairs.txt'
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

#writing a csv file
with open(path_to_savefile, 'w', encoding='utf-8') as output_file:
    writer = csv.writer(output_file, delimiter = delimiter)
    for each_pair in pairs:
        writer.writerow(each_pair)
print('Formatted file created.')

#view the created formatted text file
with open(path_to_savefile, 'rb') as file:
    lines = file.readlines()

normalized_pairs = [[process.normalize_string(s) for s in line.decode('utf8').split('\t')]for line in lines]

voc = Vocabulary()

filtered_pairs = process.filter_pairs(pairs)
print('Total number of conversations in the dataset:', len(pairs))
print('Filtered conversations:',len(filtered_pairs))

for each_pair in filtered_pairs:
    voc.add_sentence(each_pair[0])
    voc.add_sentence(each_pair[1])
print('Total words added to vocabulary:', voc.num_of_words-3)

#remove words from vocab which occur less than 3 times in all chats
#remove the pairs which contain  
trimmed_pairs = voc.trim_rare_words(filtered_pairs)
print(f'Total number of pairs: {len(filtered_pairs)},\nNumber of pairs after rare word removal: {len(trimmed_pairs)}')

#path to save trained model
save_model = SAVE_MODEL

start_training(voc, trimmed_pairs, save_model, load_model_path)
