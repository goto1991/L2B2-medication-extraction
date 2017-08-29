import random


class Iterator():
    def __init__(self, inputs, labels):
        self.texts = inputs
        self.labels = labels
        self.size = len(self.texts)
        self.epochs = 0
        self.cursor = 0
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


class RNNIterator():
    def __init__(self, texts, labels, lengths):
        self.texts = texts
        self.lengths = lengths
        self.labels = labels
        self.size = len(self.texts)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        temp = list(zip(self.texts, self.lengths, self.labels))
        random.shuffle(temp)
        self.texts, self.lengths, self.labels = zip(*temp)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res_texts = self.texts[self.cursor:self.cursor+n]
        res_lengths = self.lengths[self.cursor:self.cursor+n]
        res_labels = self.labels[self.cursor:self.cursor+n]
        self.cursor += n
        return res_texts, res_labels, res_lengths


class S2SIterator():
    def __init__(self, enc_inps, enc_labs, enc_inp_lens, dec_inps, dec_inp_lens, dec_outs, target_weights):
        self.enc_inps = enc_inps
        self.enc_labs = enc_labs
        self.enc_inp_lens = enc_inp_lens
        self.dec_inps = dec_inps
        self.dec_inp_lens = dec_inp_lens
        self.dec_outs = dec_outs
        self.target_weights = target_weights
        self.size = len(self.enc_inps)
        self.epochs = 0
        self.cursor = 0
        self.shuffle()

    def shuffle(self):
        temp = list(zip(self.enc_inps, self.enc_labs, self.enc_inp_lens, self.dec_inps, self.dec_inp_lens, self.dec_outs, self.target_weights))
        random.shuffle(temp)
        self.enc_inps, self.enc_labs, self.enc_inp_lens, self.dec_inps, self.dec_inp_lens, self.dec_outs, self.target_weights = zip(*temp)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res_enc_inps = list(self.enc_inps[self.cursor:self.cursor+n])
        res_enc_labs = list(self.enc_labs[self.cursor:self.cursor+n])
        res_enc_inp_lens = list(self.enc_inp_lens[self.cursor:self.cursor + n])
        res_dec_inps = list(self.dec_inps[self.cursor:self.cursor + n])
        res_dec_inp_lens = list(self.dec_inp_lens[self.cursor:self.cursor + n])
        res_dec_outs = list(self.dec_outs[self.cursor:self.cursor + n])
        res_target_weights = list(self.target_weights[self.cursor:self.cursor + n])
        maxinlen = max(res_enc_inp_lens)
        maxoutlen = max(res_dec_inp_lens)
        for i in range (n):
            res_enc_inps[i] = res_enc_inps[i][:maxinlen]
            res_enc_labs[i] = res_enc_labs[i][:maxinlen]
            res_dec_inps[i] = res_dec_inps[i][:maxoutlen]
            res_dec_outs[i] = res_dec_outs[i][:maxoutlen]
            res_target_weights[i] = res_target_weights[i][:maxoutlen]
        self.cursor += n
        return res_enc_inps, res_enc_labs, res_enc_inp_lens, res_dec_inps, res_dec_inp_lens, res_dec_outs, res_target_weights


class ELS2SIterator():
    def __init__(self, targets, tokens, labels, lengths, outputs):
        self.targets = targets
        self.tokens = tokens
        self.labels = labels
        self.lengths = lengths
        self.outputs = outputs
        self.size = len(self.targets)
        self.epochs = 0
        self.cursor = 0
        self.shuffle()

    def shuffle(self):
        temp = list(zip(self.targets, self.tokens, self.labels, self.lengths, self.outputs))
        random.shuffle(temp)
        self.targets, self.tokens, self.labels, self.lengths, self.outputs= zip(*temp)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res_targets = list(self.targets[self.cursor:self.cursor+n])
        res_tokens = list(self.tokens[self.cursor:self.cursor+n])
        res_labels = list(self.labels[self.cursor:self.cursor + n])
        res_lengths = list(self.lengths[self.cursor:self.cursor + n])
        res_outputs = list(self.outputs[self.cursor:self.cursor + n])
        self.cursor += n
        return res_targets, res_tokens, res_labels, res_lengths, res_outputs
