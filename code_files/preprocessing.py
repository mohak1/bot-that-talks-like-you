import os
import re
import unicodedata
import codecs

class PreprocessChats:
    '''
    this class does the following:
    1. renames the downloaded whatsapp chat backups as contact names
    2. removes the auto-generated 'back to back encryption' texts
    3. removes the 'deleted' messages
    4. removes the 'media omitted' messages 
    '''
    
    def __init__(self, CHAT_FOLDER, MASTER_FOLDER, YOUR_NAME, MAX_LENGTH):
        
        #Path to the folder where chat backups are stored
        # You Are Not Required To Make Changes Or Un-Comment The Lines Below 

        # CHAT_FOLDER = 'path_to_chats_backup_folder/WhatsApp Unknown ####-##-## at ##.##.##/'
        # MASTER_FOLDER = 'path_to_repository/processed_chats/'
        # YOUR_NAME = Name that appears in your whatsapp (go to whatsapp settings for clarification)

        self.chat_folder = CHAT_FOLDER
        self.master_folder = MASTER_FOLDER
        self.your_name = YOUR_NAME
        self.max_length = MAX_LENGTH
        

    def chat_rename(self):
        #rename all the chat backup files as contact names

        file_list = os.listdir(self.chat_folder)

        for i in file_list:
            if i == '.txt' or '.txt' not in i:
              continue
            
            contact_name = i[19:].strip('.txt')
            #rename the file with as contact_name
            os.rename(self.chat_folder+i, self.chat_folder+contact_name+'.txt')

        #print completion message
        print('All chats have been renamed.')


    def remove_auto_messages(self):
        #remove autogenerated/useless information from chats

        file_list = os.listdir(self.chat_folder)
        
        for i in file_list:
            if i == '.txt' or '.txt' not in i:
              continue
            
            print('filename:',i)

            chat = ''
            f = open(self.chat_folder+i)
            chat = f.read()
            f.close()

            #remove date-time from each line
            date_time_pattern = "([0-9][0-9]\/[0-1][0-9]\/[0-9][0-9][0-9][0-9]\,\s[0-6][0-9]\:[0-9][0-9]\s\-\s)"
            chat = re.sub(date_time_pattern, '', chat)

            #remove end-to-end encryption message
            end_to_end = 'Messages to this chat and calls are now secured with end-to-end encryption. Tap for more info.'
            chat = chat.replace(end_to_end, '')
            
            #remove .txt from filename (i)
            i = i.replace('.txt','')

            #remove media-omitted message
            media_omitted = i+': <Media omitted>'
            media_omitted_you = self.your_name+': <Media omitted>'
            chat = chat.replace(media_omitted, '')
            chat = chat.replace(media_omitted_you, '')
            
            #remove deleted message
            deleted = i+': This message was deleted'
            deleted_you = self.your_name+': You deleted this message'
            chat = chat.replace(deleted, '')
            chat = chat.replace(deleted_you, '')
            print(deleted)

            #missed video/voice call
            missed_video = i+': Missed video call'
            missed_video_you = self.your_name+': Missed video call'
            missed_voice = i+': Missed voice call'
            missed_voice_you = self.your_name+': Missed voice call'
            chat = chat.replace(missed_voice, '')
            chat = chat.replace(missed_voice_you, '')
            chat = chat.replace(missed_video, '')
            chat = chat.replace(missed_video_you, '')

            #remove emogis
            emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U0001F680-\U0001F6C0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001F681-\U0001F6C5"
                           u"\U0001F30D-\U0001F567"
                           u"\U0001F550-\U0001F559"
                           "]+", flags=re.UNICODE)

            chat = emoji_pattern.sub(r'', chat) #no emoji

            #save the chats in processed_chats dir
            new_file = open(self.master_folder+i+'txt', "w+")
            new_file.write(chat)
            new_file.close()

        print('Autogenerated messages have been removed from chats.\nProcessed data has been stored in processed_chats folder.')



    def create_pairs(self):
        #creates [question, response] pairs from processed chats
        #returns a list of these pairs

        #path to processed_chats
        path_to_folder = self.master_folder
        your_name = self.your_name

        chat_files = os.listdir(path_to_folder)
        your_name = your_name + ': '
        
        #pairs of the converstaions
        chat_pairs = []

        print('Processing chats.')

        for i in chat_files:
          #ignore autogenerated files that are not chats 
            if i == '.txt' or '.txt' not in i:
                continue

            f = open(path_to_folder+i)
            chat = f.read().split('\n')
            f.close()
            #start collecting the chat pairs
            previous = ''
            #your response
            you = []
            #the question asked by the other person
            other = []
            contact_name = i.replace('.txt', '') + ': '
            
            for text in chat:
                #if this is the start of the file and the text is from you, ignore it.
                if len(previous)==0 and your_name in text:
                    continue
                #first chat from your contact
                elif contact_name in text:
                    #this is the first text from the contact in the chat
                    #remove the name of contact and append in the list 'other' list
                    #make pairs
                    if (previous==your_name):
                        #this message is from the contact and the previous message was from you
                        #make pair and clear the lists
                        #pair: question, target response
                        chat_pairs.append([other, you])
                        other = []
                        you = []
                    msg = text.replace(contact_name, '')
                    other.append(msg)
                    #update previous
                    previous = contact_name 
                elif your_name in text:
                    #this is your response text
                    msg = text.replace(your_name, '')
                    you.append(msg)
                    previous = your_name
            contact_name = contact_name.split(':')[0]
            #print(f'Chat with {contact_name} processed')
        #print(f'your chats len:{len(other)}, their chat len: {len(other)}')

        #insert a marker to denote end of process
        print('-------------------------------------------------------------------')
        print('All chats have been processed.')

        #join all the strings in each text
        for single_pair in chat_pairs:
            single_pair[0] = ', '.join(single_pair[0])
            single_pair[1] = ', '.join(single_pair[1])

        return chat_pairs

    #converting the special characters of different languages in english alphabets
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r'([.!?])', r'\1 ', s)
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s).strip()
        return s


    def filter_pair(self, pair):
        #check if length is more than 1 (to filter empty sentences)
        #and less than MAX_LENGTH
        if len(pair[0].split())>1 and len(pair[1].split())>1 and len(pair[0].split())<self.max_length and  len(pair[1].split())<self.max_length:
            return True 
        else:
            return False

    def filter_pairs(self, pairs):
        #get a list of filtered pair of conversations
        return [pair for pair in pairs if self.filter_pair(pair)]
