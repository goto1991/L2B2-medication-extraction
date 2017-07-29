import tensorflow as tf
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn as sk

from Iterator import Iterator


class RNN_Model:

    def __init__(self, vocab_size, state_size, num_classes, window, dropout=1.0, learn_rate=0.01, emb_layer=[]):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.num_classes = num_classes
        self.window = window
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
        x = tf.placeholder(tf.int32, shape=[None, self.window])  # [batch_size, num_steps]
        # seqlen = tf.constant(tf.int32, self.window)
        y = tf.placeholder(tf.int32, shape=[None])
        keep_prob = tf.constant(1.0)

        # Embedding layer
        if self.emb_layer == []:
            embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, self.state_size])
        else:
            embeddings = tf.get_variable('embedding_matrix', initializer=tf.constant(self.emb_layer))
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        # RNN
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        # init_state = tf.get_variable('init_state', [1, 2 * self.state_size], initializer=tf.constant_initializer(0.0))
        # init_state = tf.tile(init_state, [batch_size, 1])
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)

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

        last_rnn_output = tf.reduce_mean(rnn_outputs, 1)

        # Softmax layer
        with tf.variable_scope('softmax'):
            W = tf.Variable(tf.truncated_normal(shape=[self.state_size, self.num_classes], stddev=0.05))
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
        logits = tf.matmul(last_rnn_output, W) + b
        preds = tf.nn.softmax(logits)

        correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        prediction = tf.argmax(preds, 1)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        self.graph = {'x': x,
                      'y': y,
                      'keep_prob': keep_prob,
                      'loss': loss,
                      'ts': train_step,
                      'prediction': prediction,
                      'accuracy': accuracy}

    def train(self, sets, epochs=10, batch=50, report_percentage=1, show_progress=False, show_plot=False):
        # Start a tf session and run the optimisation algorithm
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        trainer = Iterator(sets['train_set'], np.argmax(sets['train_labels'], 1))

        validation_truth = np.argmax(sets['validation_labels'], 1)
        test_truth = np.argmax(sets['test_labels'], 1)

        train_feed = {self.graph['x']: sets['train_set'],
                      self.graph['y']: np.argmax(sets['train_labels'], 1),
                      self.graph['keep_prob']: 1.0}
        validation_feed = {self.graph['x']: sets['validation_set'],
                           self.graph['y']: np.argmax(sets['validation_labels'], 1),
                           self.graph['keep_prob']: 1.0}
        test_feed = {self.graph['x']: sets['test_set'],
                     self.graph['y']: np.argmax(sets['test_labels'], 1),
                     self.graph['keep_prob']: 1.0}

        train_accuracy = []
        validation_accuracy = []
        validation_f1_score = []

        mark = (epochs * (len(sets['train_set']) // batch) * report_percentage) // 100
        check_point = []
        N = 0

        warnings.simplefilter("ignore")
        while trainer.epochs < epochs:
            trd, trl = trainer.next_batch(batch)
            if N % mark == 0:
                train_accuracy.append(self.sess.run(self.graph['accuracy'], feed_dict=train_feed))
                validation_accuracy.append(self.sess.run(self.graph['accuracy'], feed_dict=validation_feed))
                prediction = self.sess.run(self.graph['prediction'], feed_dict=validation_feed)
                validation_f1_score.append(sk.metrics.f1_score(validation_truth, prediction, pos_label=0))
                check_point.append(N)
                if show_progress: print("Progress: %d%%" % (N * report_percentage // mark), end="\r")
            feed = {self.graph['x']: trd, self.graph['y']: trl, self.graph['keep_prob']: self.dropout}
            self.sess.run(self.graph['ts'], feed_dict=feed)
            N += 1
        warnings.simplefilter("default")

        if show_plot:
            np_check_point = np.array(check_point)
            np_train_accuracy = np.array(train_accuracy)
            np_test_accuracy = np.array(validation_accuracy)
            np_validation_f1_score = np.array(validation_f1_score)

            plt.plot(np_check_point, np_train_accuracy, label="Train Accuracy")
            plt.plot(np_check_point, np_test_accuracy, label="Validation Accuracy")
            plt.plot(np_check_point, np_validation_f1_score, label="Validation F1-Score")
            plt.xlabel("Batches")
            plt.ylabel("Performance Values")
            plt.legend()
            plt.show()

        test_f1_score = sk.metrics.f1_score(test_truth, self.sess.run(self.graph['prediction'], feed_dict=test_feed),
                                            pos_label=0)
        if show_progress:
            print('FInal Values: TrAcc: {:.3f}, ValAcc: {:.3f}, ValF1: {:.3f}'.format(train_accuracy[-1],
                                                                                      validation_accuracy[-1],
                                                                                      validation_f1_score[-1]))
            print("Test F1-Score: {:.3f}\n".format(test_f1_score))
        return train_accuracy, validation_accuracy, validation_f1_score, test_f1_score