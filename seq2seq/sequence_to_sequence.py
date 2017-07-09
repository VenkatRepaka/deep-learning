import helper
import tensorflow as tf
from distutils.version import LooseVersion
import numpy as np

source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

source_sentences[:50].split('\n')
target_sentences[:50].split('\n')


def extract_character_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    set_words = set([character for line in data.split('\n') for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

# Build int2letter and letter2int dicts
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)

# Convert characters to ids
source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<unk>']) for letter in line] for line in source_sentences.split('\n')]
target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<unk>']) for letter in line] for line in target_sentences.split('\n')]

print("Example source sequence")
print(source_letter_ids[:3])
print("\n")
print("Example target sequence")
print(target_letter_ids[:3])


def pad_id_sequences(source_ids, source_letter_to_int, target_ids, target_letter_to_int, sequence_length):
    new_source_ids = [word + [source_letter_to_int['<pad>']]*(sequence_length-len(word)) for word in source_ids]
    new_target_ids = [word + [target_letter_to_int['<pad>']] * (sequence_length - len(word)) for word in target_ids]
    return new_source_ids, new_target_ids

sequence_length = max([len(word) for word in source_letter_ids] + [len(word) for word in target_letter_ids])
source_ids, target_ids = pad_id_sequences(source_letter_ids, source_letter_to_int, target_letter_ids, target_letter_to_int, sequence_length)

print("Sequence Length")
print(sequence_length)
print("\n")
print("Input sequence example")
print(source_ids[:3])
print("\n")
print("Target sequence example")
print(target_ids[:3])

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 13
decoding_embedding_size = 13
# Learning Rate
learning_rate = 0.001

input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
lr = tf.placeholder(tf.float32)

source_vocab_size = len(source_letter_to_int)
enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

enc_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
enc_rnn_cell = tf.contrib.rnn.MultiRNNCell([enc_cell]*num_layers)
_, enc_state = tf.nn.dynamic_rnn(enc_rnn_cell, enc_embed_input, dtype=tf.float32)

ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
dec_input = tf.concat([tf.fill([batch_size, 1], target_letter_to_int['s']), ending], 1)
demonstration_outputs = np.reshape(range(batch_size*sequence_length), (batch_size, sequence_length))

sess = tf.InteractiveSession()
print("Targets")
print(demonstration_outputs[:2])
print("\n")
print("Processed Decoding Input")
print(sess.run(dec_input, {targets: demonstration_outputs})[:2])

target_vocab_size = len(target_letter_to_int)

dec_embedding = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
dec_embed_input = tf.nn.embedding_lookup(dec_embedding, dec_input)

dec_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
dec_cell_rnn = tf.contrib.rnn.MultiRNNCell([dec_cell]*num_layers)

with tf.variable_scope('decoding') as decoding_scope:
    output_fn = lambda x: tf.contrib.layers.fully_connected(x, target_vocab_size, None, scope=decoding_scope)

with tf.variable_scope('decoding') as decoding_scope:
    train_decoder_fun = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
    train_prediction, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell_rnn, train_decoder_fun, dec_embed_input, sequence_length, scope=decoding_scope)

    train_logits = output_fn(train_prediction)

with tf.variable_scope('decoding', reuse=True) as decoding_scope:
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, enc_state, dec_embedding, target_letter_to_int.get('<s>'), target_letter_to_int.get('<\s>'), sequence_length-1, target_vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell_rnn, infer_decoder_fn, scope=decoding_scope)

cost = tf.contrib.seq2seq.sequence_loss(train_logits, targets, tf.ones([batch_size, sequence_length]))
optimizer = tf.train.AdamOptimizer(lr)

gradients = optimizer.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(capped_gradients)

train_source = source_ids[batch_size:]
train_target = target_ids[batch_size:]

valid_source = source_ids[:batch_size]
valid_target = target_ids[:batch_size]

sess.run(tf.global_variables_initializer())

for epoch_i in range(epochs):
    for batch_i, (source_batch, target_batch) in enumerate(helper.batch_data(train_source, train_target, batch_size)):
        _, loss = sess.run([train_op, cost], {input_data: source_batch, targets: target_batch, lr: learning_rate})
        batch_train_logits = sess.run(inference_logits, {input_data: source_batch})
        batch_valid_logits = sess.run(inference_logits, {input_data: valid_source})
        train_accuracy = np.mean(np.equal(target_batch, np.argmax(batch_train_logits, 2)))
        valid_accuracy = np.mean(np.equal(valid_target, np.argmax(batch_valid_logits, 2)))
        print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy : {:6.3f}, Validation Accuracy : {:6.3f}, Loss : {:6.3f}'
                .format(epoch_i, batch_i, len(source_ids) // batch_size, train_accuracy, valid_accuracy, loss))

input_word = 'Hello'
input_word = [source_letter_to_int.get(letter, source_letter_to_int['<unk>']) for letter in input_word.lower()]
input_word = input_word + [0] * (sequence_length - len(input_word))
batch_shell = np.zeros((batch_size, sequence_length))
batch_shell[0] = input_word
chatbot_logits = sess.run(inference_logits, {input_data: batch_shell})

print('Input')
print('  Word Ids:      {}'.format([i for i in input_word]))
print('  Input Words:   {}'.format([source_letter_to_int[i] for i in input_word]))

print('Output')
print('  Word Ids:      {}'.format([i for i in np.argmax(chatbot_logits, 1)]))
print('  Chatbot Answer Words {}'.format([target_letter_to_int[i] for i in np.argmax(chatbot_logits, 1)]))
