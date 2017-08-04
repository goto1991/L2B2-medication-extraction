import tensorflow as tf
import numpy as np
import warnings
import sklearn as sk
import matplotlib.pyplot as plt
import os
import shutil

from Iterator import Iterator


class FF_Model:
    def __init__(self, input_size=100, output_size=2, layers=[50], dropout=1.0, learn_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.weights = []
        self.biases = []
        self.hiddens = []
        self.graph = None
        self.sess = None

    def build_graph(self):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.05)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        if self.sess:
            self.close()
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        y_ = tf.placeholder(tf.float32, shape=[None, self.output_size])

        # Define the first layer here
        self.weights = [weight_variable([self.input_size, self.layers[0]])]
        self.biases = [bias_variable([self.layers[0]])]
        self.hiddens = [tf.nn.sigmoid(tf.matmul(x, self.weights[0]) + self.biases[0])]

        for i in range(1, len(self.layers)):
            self.weights.append(weight_variable([self.layers[i - 1], self.layers[i]]))
            self.biases.append(bias_variable([self.layers[i]]))
            self.hiddens.append(tf.nn.sigmoid(tf.matmul(self.hiddens[i - 1], self.weights[i]) + self.biases[i]))

        # Use dropout for this layer
        keep_prob = tf.placeholder(tf.float32)
        h_drop = tf.nn.dropout(self.hiddens[-1], keep_prob)

        # Define the output layer here
        V = weight_variable([self.layers[-1], self.output_size])
        c = bias_variable([self.output_size])
        y = tf.nn.softmax(tf.matmul(h_drop, V) + c)

        # We'll use the cross entropy loss function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

        # And classification accuracy with F1-Score
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred = tf.argmax(y, 1)

        # And the Adam optimiser
        train_step = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(cross_entropy)

        self.graph = {
            'x': x,
            'y_': y_,
            'keep_prob': keep_prob,
            'accuracy': accuracy,
            'prediction': pred,
            'ts': train_step
        }

    def train(self, sets, epochs=10, batch=50, report_percentage=1, show_progress=False, show_plot=False):
        # Start a tf session and run the optimisation algorithm
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        trainer = Iterator(sets['train_set'], sets['train_labels'])

        validation_truth = np.argmax(sets['validation_labels'], 1)
        test_truth = np.argmax(sets['test_labels'], 1)

        train_feed = {self.graph['x']: sets['train_set'],
                      self.graph['y_']: sets['train_labels'],
                      self.graph['keep_prob']: 1.0}
        validation_feed = {self.graph['x']: sets['validation_set'],
                           self.graph['y_']: sets['validation_labels'],
                           self.graph['keep_prob']: 1.0}
        test_feed = {self.graph['x']: sets['test_set'],
                     self.graph['y_']: sets['test_labels'],
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
            feed = {self.graph['x']: trd, self.graph['y_']: trl, self.graph['keep_prob']: self.dropout}
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

        test_f1_score = sk.metrics.f1_score(test_truth, self.sess.run(self.graph['prediction'], feed_dict=test_feed), pos_label=0)
        if show_progress:
            print('FInal Values: TrAcc: {:.4f}, ValAcc: {:.4f}, ValF1: {:.4f}'.format(train_accuracy[-1], validation_accuracy[-1], validation_f1_score[-1]))
            print("Test F1-Score: {:.4f}\n".format(test_f1_score))
        return train_accuracy, validation_accuracy, validation_f1_score, test_f1_score

    def predict(self, data):
        dummy = [[1, 1] for i in range(len(data))]
        feed = {self.graph['x']: data, self.graph['y_']: dummy, self.graph['keep_prob']: 1.0}
        return self.sess.run(self.graph['prediction'], feed_dict=feed)

    def close(self):
        self.sess.close()

    def save_model(self, path='tmp'):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
        os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(self.sess, path + '/model.ckpt')
        f = open(path + '/parameters', 'w+')
        f.write(str(self.input_size) + '\n')
        f.write(str(self.output_size) + '\n')
        f.write(str(self.dropout) + '\n')
        f.write(str(self.learn_rate) + '\n')
        f.write(' '.join(str(layer) for layer in self.layers) + '\n')
        f.close()
        print('Model Saved to {}'.format(path))

    def load_model(self, path='tmp'):
        f = open(path + '/parameters', 'r')
        self.input_size = int(f.readline())
        self.output_size = int(f.readline())
        self.dropout = float(f.readline())
        self.learn_rate = float(f.readline())
        self.layers = list(map(int, f.readline().split()))
        f.close()
        self.build_graph()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, path + '/model.ckpt')
