import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from data_loader import DataLoader
from common import config as cfg
from pr import precision_and_recall
from rouge import rouge


# tf.enable_eager_execution()
print('*** Tensorflow executing eagerly:', tf.executing_eagerly(), '\n')

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

data_loader = DataLoader()

round_nums = 100
num_steps = 1000

beam_width = 50

SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

def seq2seq(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']

    inp = features['encoder_inputs']
    decoder_output = features['decoder_outputs']
    decoder_input = features['decoder_inputs']

    batch_size = tf.shape(inp)[0]
    output_max_length = tf.shape(decoder_output)[1]

    start_tokens = tf.to_int32(tf.fill([batch_size], SOS_TOKEN))
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), decoder_input], 1)

    input_lengths = tf.count_nonzero(inp, 1, dtype=tf.int32)
    output_lengths = tf.count_nonzero(train_output, 1, dtype=tf.int32)

    embeddings = tf.get_variable('embeddings', [config.num_words, embed_dim])

    input_embed = tf.nn.embedding_lookup(embeddings, inp)
    output_embed = tf.nn.embedding_lookup(embeddings, train_output)

    cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, input_lengths)

    def dec_cell(encoder_outputs, input_lengths):
        attention = tf.contrib.seq2seq.BahdanauAttention(
            num_units = num_units,
            memory = encoder_outputs,
            memory_sequence_length = input_lengths)

        wrapper = tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.contrib.rnn.GRUCell(num_units=num_units),
            attention_mechanism = attention,
            attention_layer_size = num_units)

        return tf.contrib.rnn.OutputProjectionWrapper(wrapper, vocab_size)

    with tf.variable_scope('decoding_scope'):
        if mode == 'train':
            cell = dec_cell(encoder_outputs, input_lengths)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=train_helper,
                                                      initial_state=cell.zero_state(dtype=tf.float32,
                                                                                    batch_size=batch_size))

            train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                    output_time_major=False,
                                                                    impute_finished=True,
                                                                    maximum_iterations=output_max_length)

        else:
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(input_lengths, multiplier=beam_width)

            cell = dec_cell(tiled_encoder_outputs, tiled_sequence_length)

            decoder_initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, embeddings,
                                                           start_tokens=start_tokens,
                                                           end_token=EOS_TOKEN,
                                                           initial_state=decoder_initial_state,
                                                           beam_width=beam_width,
                                                           length_penalty_weight=0.0)

            pred_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                   output_time_major=False,
                                                                   impute_finished=False,
                                                                   maximum_iterations=5)

    if mode == 'train':
        weights = tf.to_float(tf.sign(decoder_output))

        loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, decoder_output, weights=weights)

        optimizer = tf.train.AdamOptimizer(1e-3)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, .1)
        optimize = optimizer.apply_gradients(zip(gradients, variables), tf.train.get_global_step())

        print('***', tf.global_norm(gradients))

        #train_op = layers.optimize_loss(loss_op, tf.train.get_global_step(),
        #                                optimizer=tf.train.AdamOptimizer(),
        #                                learning_rate=params.get('learning_rate', 0.001),
        #                                summaries=['loss', 'learning_rate'])

        return tf.estimator.EstimatorSpec(mode=mode, predictions=None,
                                          loss=loss, train_op=optimize)

    else:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.predicted_ids)

def train_seq2seq(input_filename, output_filename, model_dir):
    params = {
        'vocab_size': config.num_words,
        'batch_size': 32,
        'input_max_length': config.input_max_length,
        'output_max_length': config.output_max_length,
        'embed_dim': 150,
        'num_units': 256
    }

    def input_fn(input_filename, output_filename, batch_size, shuffle_buffer=1):

        encoder_input_data_gen = lambda: data_loader.data_generator_3(input_filename, is_encoder_input=True)
        decoder_output_data_gen = lambda: data_loader.data_generator_3(output_filename)
        decoder_input_data_gen = lambda: data_loader.data_generator_3(output_filename, is_decoder_input=True)

        encoder_input_data = tf.data.Dataset.from_generator(encoder_input_data_gen,
                                                            output_types=tf.int32,
                                                            output_shapes=(None,))
        decoder_output_data = tf.data.Dataset.from_generator(decoder_output_data_gen,
                                                             output_types=tf.int32,
                                                             output_shapes=(None,))
        decoder_input_data = tf.data.Dataset.from_generator(decoder_input_data_gen,
                                                            output_types=tf.int32,
                                                            output_shapes=(None,))

        dataset = tf.data.Dataset.zip((encoder_input_data, decoder_output_data, decoder_input_data)).shuffle(shuffle_buffer).repeat(1).padded_batch(batch_size,
                                                                                                                                                    padded_shapes=([None],[None],[None]))

        iterator = dataset.make_one_shot_iterator()

        encoder_inputs, decoder_outputs, decoder_inputs = iterator.get_next()

        return {'encoder_inputs': encoder_inputs, 'decoder_outputs': decoder_outputs, 'decoder_inputs': decoder_inputs}


    est = tf.estimator.Estimator(model_fn=seq2seq,
                                 model_dir=model_dir,
                                 params=params)

    train_input_func = lambda: input_fn(config.source_data_train, config.target_data_train, params['batch_size'], shuffle_buffer=1000)
    eval_input_func = lambda: input_fn(config.source_data_dev, config.target_data_dev, params['batch_size'])
    test_input_func = lambda: input_fn(config.source_data_test, config.target_data_test, params['batch_size'])

    est.train(input_fn=train_input_func, steps=20000)
    for r in range(num_rounds):
        # training for num_steps steps
        print('\nRound', r + 1)
        est.train(input_fn=train_input_func, steps=num_steps)
        
        # evaluatation
        print('\nEvaluation:')
        predictions = est.predict(input_fn=test_input_func)

        # writing the predictions into a file
        print('\n\nWriting Predictions...')
        for i, pred in enumerate(predictions):
            with open('./predictions/' + str(i), 'w+') as pred_file:
                for keyph in np.array(pred).T:
                    pred_file.write(data_loader.index_to_sent(keyph).replace('<EOS>', '')
                                                                    .replace('<UNK>', '')
                                                                    .replace('<SOS>', '') + '\n')
                    
        # running the evaluation metrics, precision, recall, f1-score, and ROUGE
        precision_and_recall(r)
        rouge(5)
        rouge(10)

def main():
    train_seq2seq('input', 'output', 'model/seq2seq')

if __name__ == '__main__':
    main()
