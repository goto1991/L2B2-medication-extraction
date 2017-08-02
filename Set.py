import re
import os
import numpy as np

import Functions as fn
from DS import DS


class pool:
    def __init__(self, data=[]):
        self.data = data
        self.size = len(data)

    def present(self, item):
        for case in self.data:
            if item.raw_text == case.raw_text:
                return True
        return False

    def add(self, item):
        if not self.present(item):
            self.data.append(item)
            self.size += 1
            return 1
        return 0

    def add_labels(self, name, case, raw_labels):
        for i in range(self.size):
            if self.data[i].name == name:
                self.data[i].labelled = 'yes'
                self.data[i].label_type = case
                self.data[i].raw_labels = raw_labels
                break

    def write_texts(self, path):
        os.makedirs(path)
        for i in range(self.size):
            f = open(path + '/' + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
            f.write(self.data[i].raw_text)
            f.close()
        print('raw_text Write Complete')

    def write_labels(self, path):
        os.makedirs(path)
        for i in range(self.size):
            if self.data[i].labelled == 'yes':
                f = open(path + '/' + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
                f.write(self.data[i].raw_labels)
                f.close()
        print('raw_labels Write Complete')

    def load_texts(self, path):
        for filename in fn.listdir_nohidden(path):
            info = filename.split('_')
            with open(path + '/' + filename, 'r') as file:
                temp = DS(name=info[0], challenge=info[1], stage=info[2], raw_text=file.read())
                self.add(temp)
        print('Raw Text Load Complete')

    def load_labels(self, path):
        for filename in fn.listdir_nohidden(path):
            info = filename.split('_')
            with open(path + '/' + filename, 'r') as file:
                self.add_labels(name=info[0], case=info[4], raw_labels=file.read())
        print('Raw Labels Load Complete')

    def number_of(self, challenge=r'.', stage=r'.', labelled=r'.', label_type=r'.'):
        n = 0
        for case in self.data:
            if (re.match(challenge, case.challenge) != None) & \
                    (re.match(stage, case.stage) != None) & \
                    (re.match(labelled, case.labelled) != None) & \
                    (re.match(label_type, case.label_type) != None):
                n += 1
        return (n)

    def get_DS(self, name=r'.', challenge=r'.', stage=r'.', labelled=r'.', label_type=r'.'):
        output = pool([])
        for case in self.data:
            if (re.match(name, case.name) != None) & \
                    (re.match(challenge, case.challenge) != None) & \
                    (re.match(stage, case.stage) != None) & \
                    (re.match(labelled, case.labelled) != None) & \
                    (re.match(label_type, case.label_type) != None):
                output.add(case)
        return output

    def show_info(self):
        for case in self.data:
            case.show_info()
            print('\n')

    def append(self, dataset):
        for case in dataset:
            self.add(case)

    def duplicates(self):
        dupl = 0
        for i in range(len(self.data)):
            occurrence = 0
            for j in range(i, len(self.data)):
                if self.data[i].emb_text == self.data[j].emb_text:
                    occurrence += 1
                    if occurrence > 1:
                        occurrence = 1
                        dupl += 1
        return dupl

    def example_duplicate(self):
        for i in range(len(self.data)):
            occurrence = 0
            for j in range(i, len(self.data)):
                if self.data[i].emb_text == self.data[j].emb_text:
                    occurrence += 1
                    if occurrence > 1:
                        return self.data[i], self.data[j]

    def process_for_embedding(self):
        for i in range(self.size):
            self.data[i].process_for_embedding()

    def get_sentences(self, challenge=r'.', stage=r'.'):
        sentences = []
        temp = self.get_DS(challenge=challenge, stage=stage)
        for case in temp.data:
            for sent in case.emb_text:
                sentences.append(sent)
        return sentences

    def process_for_testing(self):
        for i in range(self.size):
            self.data[i].process_for_testing()

    def get_ff_sets(self, model, left_words=0, right_words=0):
        padded_texts = [['<pad>' for i in range(left_words)] + case.test_text + ['<pad>' for i in range(right_words)]
                        for case in self.data]

        vectorized_texts = []
        for text in padded_texts:
            for i in range(left_words, len(text) - right_words):
                word_vector = [text[i + pos] for pos in range(-left_words, right_words + 1)]
                word_vector = np.concatenate(
                    [model[word] if word in model.wv.vocab else np.zeros(model.vector_size) for word in word_vector])
                vectorized_texts.append(word_vector)

        label_set = [label for case in self.data for label in case.test_labels]
        word_set = [word for case in self.data for word in case.test_text]

        return vectorized_texts, label_set, word_set

    def get_rnn_sets(self, word_indices, left_words=0, right_words=0):
        padded_texts = [['<pad>' for i in range(left_words)] + case.test_text + ['<pad>' for i in range(right_words)]
                        for case in self.data]

        sequence_set = []
        for text in padded_texts:
            for i in range(left_words, len(text) - right_words):
                word_sequence = [text[i + pos] for pos in range(-left_words, right_words + 1)]
                word_sequence = [word_indices[word] if word in word_indices.keys() else 1 for word in word_sequence]
                sequence_set.append(word_sequence)

        label_set = [label for case in self.data for label in case.test_labels]
        word_set = [word for case in self.data for word in case.test_text]

        return sequence_set, label_set, word_set
