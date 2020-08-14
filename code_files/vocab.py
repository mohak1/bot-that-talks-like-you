import itertools 

class Vocabulary:
    '''
    This class creates the vocabulary dictionary using the chat pairs
    '''
    
    def __init__(self):
        self.PAD_token = 0   #for padding the sentences for achieving same length throughout
        self.START_token = 1   #denotes the start of sentence
        self.END_token = 2   #denotes the end of the sentence
        self.word2index = {}  #word is key, index is value
        self.word2count = {}  #word is key, its frequency is value
        self.index2word = {self.PAD_token: 'PAD', self.START_token: 'START', self.END_token: 'END'}  #index is key, word is value
        self.num_of_words = 3   #denotes the total number of words in the vocabulary. Currently there are only 3: PAD, START, END
        self.keep_words = []   #list which keeps the words which have the frequency more than min_count
        MIN_COUNT = 3   #used for rare words trimming
        self.min_count = MIN_COUNT
    
    def add_sentence(self, sentence):
        #extracts all the words from a sentence and adds them to vocabulary 
        for word in sentence.split(' '):
          self.add_word(word)

    def add_word(self, word):
        #adds the word to the vocabulary by giving it an index and sets the count to 1 if its a new word
        #else, increases the count of the word, denoting the increase in frequency of that word
        if word not in self.word2index:
          self.word2index[word] = self.num_of_words
          self.word2count[word] = 1
          self.index2word[self.num_of_words] = word
          self.num_of_words += 1
        else:
          self.word2count[word] += 1

    def trim(self, min_count):
        #the vocabulary is checked and the words are removed which have a frequency lower than min_count
        for k, v in self.word2count.items():
          if v >= min_count:
            self.keep_words.append(k)
        print(f'keep words {len(self.keep_words)} out of {len(self.word2index)} words')

        #reinitialize the dictionaries to have only the trimmed words (above the min_count)
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: 'PAD', self.START_token: 'START', self.END_token: 'END'}
        self.num_of_words = 3

        for word in self.keep_words:
          self.add_word(word)

    def trim_rare_words(self, word_pairs):
        #remove the words from vocabulary which have frequency lower than MIN_WORDS
        self.trim(self.min_count)
        #remove the pairs which contain these rare words form fitered_pairs 
        keep_pairs = []
        for pair in word_pairs:
            input_sentence = pair[0]
            response_sentence = pair[1]
            #if a rare word is present in input or response, the pair is to be removed
            keep_input = True
            keep_response = True
            #loop over all the words in the input sentence
            for word in input_sentence.split(' '):
                #check if the word is present in vocab ie non-rare
                if word not in self.word2index:
                  #if a rare word is found in the sentence, it is to be removed 
                  keep_input = False
                  break  
            #loop over all the response sentences
            for word in response_sentence.split(' '):
                #check if all the words in response sentence are non-rare (present in vocab)
                if word not in self.word2index:
                    keep_response = False
                    break
            if keep_input and keep_response:
                keep_pairs.append(pair)
        return keep_pairs

    def index_from_sentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')] + [self.END_token]

    def zero_padding(self, index_matrix, fill_value=0):
        return list(itertools.zip_longest(*index_matrix, fillvalue = fill_value))

    def binary_matrix(self, index_matrix, value=0):
        matrix=[]
        for i, seq in enumerate(index_matrix):
            matrix.append([])
            for token in seq:
                if token == self.PAD_token:
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)
        return matrix