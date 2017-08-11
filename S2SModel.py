import tensorflow as tf
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn as sk

from Iterator import RNNIterator


class S2S_Model:

    def __init__(self, enc_sym, dec_sym, emb_size, state_size, num_classes, dropout=1.0, learn_rate=0.01, emb_layer=False):
        self.enc_sym = enc_sym
        self.dec_sym = dec_sym
        self.emb_size = emb_size
        self.state_size = state_size
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.emb_layer = emb_layer
        self.graph = None
        self.sess = None

    def reset_graph(self):
        if self.sess:
            self.sess.close()
        tf.reset_default_graph()

    def build_graph(self):
        self.reset_graph()

        # Placeholders
        x = tf.placeholder(tf.int32, shape=[None, None])  # [batch_size, num_steps]
        seqlen = tf.placeholder(tf.int32, shape=[None])
        y = tf.placeholder(tf.int32, shape=[None])
        keep_prob = tf.placeholder(tf.float32)

        # Embedding layer
        if type(self.emb_layer) != bool:
            embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, self.state_size])
        else:
            embeddings = tf.get_variable('embedding_matrix', initializer=tf.constant(self.emb_layer))
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        # RNN
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        drop_wrap = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(drop_wrap, rnn_inputs, sequence_length=seqlen, dtype=tf.float32)

        # Add dropout, as the model otherwise quickly overfits
        # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

        """
        Obtain the last relevant output. The best approach in the future will be to use:

            last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

        which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
        gradient for this op has not been implemented as of this writing.

        The below solution works, but throws a UserWarning re: the gradient.
        """
        # idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
        # last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

        # last_rnn_output = tf.reduce_mean(rnn_outputs, 1)

        idx = tf.range(tf.shape(seqlen)[0])*tf.shape(rnn_outputs)[1] + (seqlen - 1)
        last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, self.state_size]), idx)

        # Softmax layer
        with tf.variable_scope('softmax'):
            W = tf.Variable(tf.truncated_normal(shape=[self.state_size, self.num_classes], stddev=0.05))
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
        logits = tf.matmul(last_rnn_output, W) + b
        preds = tf.nn.softmax(logits)

        prediction = tf.argmax(preds, 1)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        self.graph = {'x': x,
                      'y': y,
                      'seqlen': seqlen,
                      'keep_prob': keep_prob,
                      'loss': loss,
                      'ts': train_step,
                      'prediction': prediction}

    def train(self, sets, epochs=10, batch=50, report_percentage=1, show_progress=False, show_plot=False):
        # Start a tf session and run the optimisation algorithm
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        trainer = RNNIterator(sets['train_set'], np.argmax(sets['train_labels'], 1), sets['train_lengths'])

        train_truth = np.argmax(sets['train_labels'], 1)
        validation_truth = np.argmax(sets['validation_labels'], 1)
        test_truth = np.argmax(sets['test_labels'], 1)

        train_feed = {self.graph['x']: sets['train_set'],
                      self.graph['y']: np.argmax(sets['train_labels'], 1),
                      self.graph['seqlen']: sets['train_lengths'],
                      self.graph['keep_prob']: 1.0}
        validation_feed = {self.graph['x']: sets['validation_set'],
                           self.graph['y']: np.argmax(sets['validation_labels'], 1),
                           self.graph['seqlen']: sets['validation_lengths'],
                           self.graph['keep_prob']: 1.0}
        test_feed = {self.graph['x']: sets['test_set'],
                     self.graph['y']: np.argmax(sets['test_labels'], 1),
                     self.graph['seqlen']: sets['test_lengths'],
                     self.graph['keep_prob']: 1.0}

        train_f1_score = []
        validation_f1_score = []

        mark = (epochs * (len(sets['train_set']) // batch) * report_percentage) // 100
        check_point = []
        N = 0

        warnings.simplefilter("ignore")
        while trainer.epochs < epochs:
            trd, trl, trle = trainer.next_batch(batch)
            if N % mark == 0:
                prediction = self.sess.run(self.graph['prediction'], feed_dict=train_feed)
                train_f1_score.append(sk.metrics.f1_score(train_truth, prediction, pos_label=0))
                prediction = self.sess.run(self.graph['prediction'], feed_dict=validation_feed)
                validation_f1_score.append(sk.metrics.f1_score(validation_truth, prediction, pos_label=0))
                check_point.append(N)
                if show_progress: print("Progress: %d%%" % (N * report_percentage // mark), end="\r")
            feed = {self.graph['x']: trd, self.graph['y']: trl, self.graph['seqlen']: trle, self.graph['keep_prob']: self.dropout}
            self.sess.run(self.graph['ts'], feed_dict=feed)
            N += 1
        warnings.simplefilter("default")

        if show_plot:
            np_check_point = np.array(check_point)
            np_train_f1 = np.array(train_f1_score)
            np_val_f1 = np.array(validation_f1_score)

            plt.plot(np_check_point, np_train_f1, label="Train")
            plt.plot(np_check_point, np_val_f1, label="Validation")
            plt.plot(np_check_point, np.ones(len(np_check_point))*0.35, label="Baseline")
            plt.xlabel("Batches")
            plt.ylabel("F1-Score")
            plt.legend()
            plt.show()

        test_f1_score = sk.metrics.f1_score(test_truth, self.sess.run(self.graph['prediction'], feed_dict=test_feed),
                                            pos_label=0)
        if show_progress:
            print('FInal Values: Tr-F1: {:.4f}, Val-F1: {:.4f}'.format(train_f1_score[-1], validation_f1_score[-1]))
            print("Test F1-Score: {:.4f}\n".format(test_f1_score))
        return train_f1_score, validation_f1_score, test_f1_score

    def predict(self, data, seqlen):
        dummy = [1 for i in range(len(data))]
        feed = {self.graph['x']: data, self.graph['y']: dummy, self.graph['seqlen']: seqlen, self.graph['keep_prob']: 1.0}
        return self.sess.run(self.graph['prediction'], feed_dict=feed)

    def close(self):
        self.sess.close()