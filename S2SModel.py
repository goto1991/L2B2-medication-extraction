import tensorflow as tf
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn as sk

from Iterator import S2SIterator


class S2S_Model:
    def __init__(self,
                 decay=0,
                 batch=50,
                 enc_vocab_size=100,
                 dec_vocab_size=9,
                 enc_emb_size=100,
                 dec_emb_size=100,
                 state_size=100,
                 dropout=1.0,
                 learn_rate=0.001,
                 max_gradient_norm=5,
                 enc_emb_layer=False):
        self.decay = decay
        self.batch = batch
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.enc_emb_size = enc_emb_size
        self.dec_emb_size = dec_emb_size
        self.state_size = state_size
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.max_gradient_norm = max_gradient_norm
        self.enc_emb_layer = enc_emb_layer
        self.graph = None
        self.sess = None

    def reset_graph(self):
        if self.sess:
            self.sess.close()
        tf.reset_default_graph()

    def build_graph(self):
        self.reset_graph()

        # Placeholders
        enc_x = tf.placeholder(name='encoder_inputs', dtype=tf.int32,
                               shape=[self.batch, None])  # [batch_size, num_steps]
        enc_labs = tf.placeholder(dtype=tf.float32, shape=[self.batch, None, 1])
        dec_x = tf.placeholder(name='decoder_inputs', dtype=tf.int32, shape=[self.batch, None])
        y = tf.placeholder(name='target', dtype=tf.int32, shape=[self.batch, None])
        target_weights = tf.placeholder(tf.float32, shape=[self.batch, None])
        enc_seqlen = tf.placeholder(tf.int32, shape=[self.batch])
        dec_seqlen = tf.placeholder(tf.int32, shape=[self.batch])
        keep_prob = tf.placeholder(tf.float32)
        testing = tf.placeholder(tf.bool)

        # Embeddings
        if type(self.enc_emb_layer) != bool:
            enc_embeddings = tf.get_variable('enc_embedding_matrix', [self.enc_vocab_size, self.enc_emb_size],
                                             dtype=tf.float32)
        else:
            enc_embeddings = tf.get_variable('enc_embedding_matrix', initializer=tf.constant(self.enc_emb_layer))
        enc_inputs = tf.nn.embedding_lookup(enc_embeddings, enc_x)
        dec_inputs = tf.nn.embedding_lookup(enc_embeddings, dec_x)
        enc_inputs = tf.concat([enc_inputs, enc_labs], axis=-1)

        # Encoder

        # Singl Forward
        # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)

        # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
        #                                                   inputs=enc_inputs,
        #                                                   sequence_length=enc_seqlen,
        #                                                   dtype=tf.float32)

        # Bidirectional
        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                                    backward_cell,
                                                                    enc_inputs,
                                                                    sequence_length=enc_seqlen,
                                                                    dtype=tf.float32)

        # Conbining the output hidden states of ceells
        encoder_outputs = tf.concat(bi_outputs, -1)
        encoder_state = tf.concat(encoder_state, axis=-1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state[0], encoder_state[1])

        # Decoder
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * self.state_size)

        # Attention
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(2 * self.state_size,
                                                                encoder_outputs,
                                                                memory_sequence_length=enc_seqlen)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                           attention_mechanism,
                                                           attention_layer_size=self.state_size)

        attn_zero = decoder_cell.zero_state(batch_size=self.batch, dtype=tf.float32)
        encoder_state = attn_zero.clone(cell_state=encoder_state)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(dec_inputs, dec_seqlen)

        projection_layer = tf.contrib.keras.layers.Dense(self.dec_vocab_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  encoder_state,
                                                  output_layer=projection_layer)

        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch)
        wordnum = tf.to_float(tf.reduce_sum(dec_seqlen))
        ent_per_word = train_loss * self.batch / wordnum
        perplexity = tf.pow(tf.to_float(2), ent_per_word)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        # Test Helper
        test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(enc_embeddings,
                                                               tf.fill([tf.shape(enc_x)[0]], 3),
                                                               4)

        # Test Decoder
        test_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                       test_helper,
                                                       encoder_state,
                                                       output_layer=projection_layer)
        # Test Dynamic decoding
        maximum_iterations = tf.round(tf.reduce_max(enc_seqlen))
        test_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(test_decoder,
                                                               maximum_iterations=maximum_iterations)
        # prediction = test_outputs.sample_id
        prediction = tf.argmax(test_outputs.rnn_output, axis=2)

        self.graph = {'enc_x': enc_x,
                      'enc_labs': enc_labs,
                      'dec_x': dec_x,
                      'y': y,
                      'enc_seqlen': enc_seqlen,
                      'dec_seqlen': dec_seqlen,
                      'target_weights': target_weights,
                      'keep_prob': keep_prob,
                      'prediction': prediction,
                      'perplexity': perplexity,
                      'update_step': update_step
                      }

    def train(self, sets, epochs=10, report_percentage=1, show_progress=False, show_plot=False):
        # Start a tf session and run the optimisation algorithm
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        trainer = S2SIterator(*(sets['train'][:-1]))

        train_truth = [truth[:end] for (truth, end) in zip(sets['train'][5], sets['train'][4])]
        validation_truth = [truth[:l] for (truth, l) in zip(sets['validation'][5], sets['validation'][4])]
        test_truth = [truth[:l] for (truth, l) in zip(sets['test'][5], sets['test'][4])]

        train_feed = {self.graph['enc_x']: sets['train'][0],
                      self.graph['enc_labs']: sets['train'][1],
                      self.graph['enc_seqlen']: sets['train'][2],
                      self.graph['dec_x']: sets['train'][3],
                      self.graph['dec_seqlen']: sets['train'][4],
                      self.graph['y']: sets['train'][5],
                      self.graph['target_weights']: sets['train'][6],
                      self.graph['keep_prob']: 1.0}
        validation_feed = {self.graph['enc_x']: sets['validation'][0],
                           self.graph['enc_labs']: sets['validation'][1],
                           self.graph['enc_seqlen']: sets['validation'][2],
                           self.graph['dec_x']: sets['validation'][3],
                           self.graph['dec_seqlen']: sets['validation'][4],
                           self.graph['y']: sets['validation'][5],
                           self.graph['target_weights']: sets['validation'][6],
                           self.graph['keep_prob']: 1.0}
        test_feed = {self.graph['enc_x']: sets['test'][0],
                     self.graph['enc_labs']: sets['test'][1],
                     self.graph['enc_seqlen']: sets['test'][2],
                     self.graph['dec_x']: sets['test'][3],
                     self.graph['dec_seqlen']: sets['test'][4],
                     self.graph['y']: sets['test'][5],
                     self.graph['target_weights']: sets['test'][6],
                     self.graph['keep_prob']: 1.0}

        train_f1_score = []
        validation_f1_score = []

        mark = (epochs * (len(sets['train'][0]) // self.batch) * report_percentage) // 100
        check_point = []
        N = 0

        warnings.simplefilter("ignore")
        while trainer.epochs < epochs:
            trex, trel, trexl, trdx, trdxl, trdy, trtw = trainer.next_batch(self.batch)
            feed = {self.graph['enc_x']: trex,
                    self.graph['enc_labs']: trel,
                    self.graph['enc_seqlen']: trexl,
                    self.graph['dec_x']: trdx,
                    self.graph['dec_seqlen']: trdxl,
                    self.graph['y']: trdy,
                    self.graph['target_weights']: trtw,
                    self.graph['keep_prob']: self.dropout}
            if N % mark == 0:
                # prediction = self.sess.run(self.graph['prediction'], feed_dict=train_feed)
                # pred_cut = [pred[:end] for (pred, end) in zip(prediction, sets['train'][3])]
                # f1_sum = 0
                # for i in range(len(pred_cut)):
                #    f1_sum += sk.metrics.f1_score(train_truth[i], pred_cut[i], labels=[1, 2, 3, 4, 5, 6], average='micro')
                # train_f1_score.append(f1_sum / len(pred_cut))
                # prediction = self.sess.run(self.graph['prediction'], feed_dict=validation_feed)
                # pred_cut = [pred[:end] for (pred, end) in zip(prediction, sets['validation'][3])]
                # f1_sum = 0
                # for i in range(len(pred_cut)):
                #    f1_sum += sk.metrics.f1_score(validation_truth[i], pred_cut[i], labels=[1, 2, 3, 4, 5, 6], average='micro')
                # validation_f1_score.append(f1_sum / len(pred_cut))
                # check_point.append(N)
                perp = self.sess.run(self.graph['perplexity'], feed_dict=feed)
                print('Epoch: {}, Learn Rate: {:.7f}, Perplexity: {:.2f}'.format(trainer.epochs, self.learn_rate, perp))
                if show_progress: print("Progress: %d%%" % (N * report_percentage // mark), end="\r")
            self.sess.run(self.graph['update_step'], feed_dict=feed)
            self.learn_rate = self.learn_rate * 1 / (1 + self.decay * trainer.epochs)
            N += 1
        warnings.simplefilter("default")

        # test_prediction = self.sess.run(self.graph['prediction'], feed_dict=test_feed)
        # pred_cut = [pred[:end] for (pred, end) in zip(prediction, sets['test'][3])]
        # f1_sum = 0
        # for i in range(len(pred_cut)):
        #    f1_sum += sk.metrics.f1_score(test_truth[i], pred_cut[i], labels=[1, 2, 3, 4, 5, 6], average='micro')
        # test_f1_score = f1_sum / len(pred_cut)

        # if show_progress:
        #    print('FInal Values: Tr-F1: {:.4f}, Val-F1: {:.4f}'.format(train_f1_score[-1], validation_f1_score[-1]))
        #    print("Test F1-Score: {:.4f}\n".format(test_f1_score))

        # if show_plot:
        #    np_check_point = np.array(check_point)
        #    np_train_f1 = np.array(train_f1_score)
        #    np_val_f1 = np.array(validation_f1_score)

        #    plt.plot(np_check_point, np_train_f1, label="Train")
        #    plt.plot(np_check_point, np_val_f1, label="Validation")
        #    plt.plot(np_check_point, np.ones(len(np_check_point))*0.35, label="Baseline")
        #    plt.xlabel("Batches")
        #    plt.ylabel("F1-Score")
        #    plt.legend()
        #    plt.show()

        # return train_f1_score, validation_f1_score, test_f1_score

    def predict(self, data):
        feed = {self.graph['enc_x']: data[0],
                self.graph['enc_labs']: data[1],
                self.graph['enc_seqlen']: data[2],
                self.graph['dec_x']: data[3],
                self.graph['dec_seqlen']: data[4],
                self.graph['y']: data[5],
                self.graph['target_weights']: data[6],
                self.graph['keep_prob']: 1.0}
        return self.sess.run(self.graph['prediction'], feed_dict=feed)

    def close(self):
        self.sess.close()