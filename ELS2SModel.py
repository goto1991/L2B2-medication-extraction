import tensorflow as tf
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn as sk

from Iterator import ELS2SIterator


class ELS2S_Model:
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
        targs = tf.placeholder(dtype=tf.int32, shape=[self.batch, 14])  # [batch_size, num_steps]
        toks = tf.placeholder(dtype=tf.int32, shape=[self.batch, 94])
        labs = tf.placeholder(dtype=tf.float32, shape=[self.batch, 94, 1])
        inp_seqlen = tf.placeholder(tf.int32, shape=[self.batch])
        y = tf.placeholder(dtype=tf.int32, shape=[self.batch, 94])
        # target_weights = tf.placeholder(tf.float32, shape=[self.batch, 94])
        # keep_prob = tf.placeholder(tf.float32)

        # Embeddings
        if type(self.enc_emb_layer) != bool:
            enc_embeddings = tf.get_variable('enc_embedding_matrix', [self.enc_vocab_size, self.enc_emb_size],
                                             dtype=tf.float32)
        else:
            enc_embeddings = tf.get_variable('enc_embedding_matrix', initializer=tf.constant(self.enc_emb_layer))
        taremb = tf.nn.embedding_lookup(enc_embeddings, targs)
        tarbow = tf.reduce_sum(taremb, 1)
        tarbowlab = tf.concat([tarbow, tf.constant(1, dtype=tf.float32, shape=[self.batch, 1])], axis=-1)

        tokind = tf.nn.embedding_lookup(enc_embeddings, toks)
        rnn_inputs = tf.concat([tokind, labs], axis=-1)

        # Bidirectional
        forward_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
        backward_cell = tf.nn.rnn_cell.GRUCell(self.state_size)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                                    cell_bw=backward_cell,
                                                                    inputs=rnn_inputs,
                                                                    sequence_length=inp_seqlen,
                                                                    initial_state_fw=tarbowlab,
                                                                    initial_state_bw=tarbowlab,
                                                                    dtype=tf.float32)

        # Conbining the output hidden states of ceells
        rnn_outputs = tf.concat(bi_outputs, -1)

        with tf.variable_scope('softmax'):
            W = tf.Variable(tf.truncated_normal(shape=[2 * self.state_size, 7], stddev=0.05))
            b = tf.Variable(tf.constant(0.1, shape=[7]))
        new_rnn_shape = tf.reshape(rnn_outputs, shape=[-1, 2 * self.state_size])
        logits = tf.matmul(new_rnn_shape, W) + b
        output = tf.reshape(logits, [self.batch, 94, 7])

        preds = tf.nn.softmax(output)
        prediction = tf.argmax(preds, 2)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
        train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        self.graph = {'targs': targs,
                      'toks': toks,
                      'labs': labs,
                      'inp_seqlen': inp_seqlen,
                      'y': y,
                      'prediction': prediction,
                      'loss': loss,
                      'ts': train_step
                      }

    def train(self, sets, epochs=10, report_percentage=1, show_progress=False, show_plot=False):
        # Start a tf session and run the optimisation algorithm
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        trainer = ELS2SIterator(*(sets['train'][:-1]))

        # train_truth = sets['train'][4]
        # validation_truth = sets['validation'][4]
        # test_truth = sets['validation'][4]

        # train_feed = {self.graph['enc_x']: sets['train'][0],
        # self.graph['enc_labs']: sets['train'][1],
        # self.graph['enc_seqlen']: sets['train'][2],
        # self.graph['dec_x']: sets['train'][3],
        # self.graph['dec_seqlen']: sets['train'][4],
        # self.graph['y']: sets['train'][5],
        # self.graph['target_weights']: sets['train'][6],
        # self.graph['keep_prob']: 1.0}
        # validation_feed = {self.graph['enc_x']: sets['validation'][0],
        # self.graph['enc_labs']: sets['validation'][1],
        # self.graph['enc_seqlen']: sets['validation'][2],
        # self.graph['dec_x']: sets['validation'][3],
        # self.graph['dec_seqlen']: sets['validation'][4],
        # self.graph['y']: sets['validation'][5],
        # self.graph['target_weights']: sets['validation'][6],
        # self.graph['keep_prob']: 1.0}
        # test_feed = {self.graph['enc_x']: sets['test'][0],
        # self.graph['enc_labs']: sets['test'][1],
        # self.graph['enc_seqlen']: sets['test'][2],
        # self.graph['dec_x']: sets['test'][3],
        # self.graph['dec_seqlen']: sets['test'][4],
        # self.graph['y']: sets['test'][5],
        # self.graph['target_weights']: sets['test'][6],
        # self.graph['keep_prob']: 1.0}

        # train_f1_score = []
        # validation_f1_score = []

        mark = (epochs * (len(sets['train'][0]) // self.batch) * report_percentage) // 100
        check_point = []
        N = 0

        warnings.simplefilter("ignore")
        while trainer.epochs < epochs:
            trtar, trtok, trlab, trlen, trout = trainer.next_batch(self.batch)
            feed = {self.graph['targs']: trtar,
                    self.graph['toks']: trtok,
                    self.graph['labs']: trlab,
                    self.graph['inp_seqlen']: trlen,
                    self.graph['y']: trout}
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
                loss = self.sess.run(self.graph['loss'], feed_dict=feed)
                print('Epoch: {}, Learn Rate: {:.7f}, Perplexity: {:.2f}'.format(trainer.epochs, self.learn_rate, loss))
                if show_progress: print("Progress: %d%%" % (N * report_percentage // mark), end="\r")
            self.sess.run(self.graph['ts'], feed_dict=feed)
            # self.learn_rate = self.learn_rate * 1/(1 + self.decay * trainer.epochs)
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
        feed = {self.graph['targs']: data[0],
                self.graph['toks']: data[1],
                self.graph['labs']: data[2],
                self.graph['y']: data[4],
                self.graph['inp_seqlen']: data[3]}
        return self.sess.run(self.graph['prediction'], feed_dict=feed)

    def close(self):
        self.sess.close()