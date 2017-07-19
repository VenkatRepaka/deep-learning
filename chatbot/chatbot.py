import os

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from keras.layers.embeddings import Embedding
from functools import reduce
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            sub_story = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                sub_story = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                sub_story = [x for x in story if x]
            data.append((sub_story, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

# path = os.path.abspath('./chatbot/tasks_1-20_v1-2.tar.gz')
# tar = tarfile.open(path)
train_stories = get_stories(open('/Users/sahdipan/deep-learning/chatbot/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'))
test_stories = get_stories(open('/Users/sahdipan/deep-learning/chatbot/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt'))

vocab = set()
for story, q, answer in train_stories+test_stories:
    vocab |= set(story+q+[answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_max_len = max(map(len, (x for x, _, _ in train_stories+test_stories)))
query_max_len = max(map(len, (x for _, x, _ in train_stories+test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_max_len, 'words')
print('Query max length:', query_max_len, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i+1) for i, c in enumerate(vocab))

inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_max_len,
                                                               query_max_len)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_max_len,
                                                            query_max_len)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

# placeholders
input_sequence = Input((story_max_len, ))
question = Input((query_max_len, ))

# encoders
# embed the input sequence into sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))

# embed the sequence into a sequence of vectors of size query_max_len
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_max_len))
input_encoder_c.add(Dropout(0.3))

# embed the questions into sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_max_len))
question_encoder.add(Dropout(0.3))

# encode input sequence and questions
# to sequence of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a match between the first input vector sequence
# and the question vector sequence
# shape: `samples, story_max_len, query_max_len`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with second input vector sequence
response = add([match, input_encoded_c])
response = Permute((2, 1))(response)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([inputs_train, queries_train], answers_train, batch_size=32, epochs=120, validation_data=([inputs_test, queries_test], answers_test))