import random


class Iterator():
    def __init__(self, inputs, labels):
        self.texts = inputs
        self.labels = labels
        self.size = len(self.texts)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        temp = list(zip(self.texts, self.labels))
        random.shuffle(temp)
        self.texts, self.labels = zip(*temp)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res_texts = self.texts[self.cursor:self.cursor+n]
        res_labels = self.labels[self.cursor:self.cursor+n]
        self.cursor += n
        return res_texts, res_labels