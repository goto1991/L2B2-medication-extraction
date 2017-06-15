import re
import os


class Set:
    def __init__(self, data=[]):
        self.data = []
        self.size = len(data)

    def present(self, item):
        for case in self.data:
            if item.raw_text == case.raw_text:
                return (True)
        return (False)

    def add(self, item):
        if not self.present(item):
            self.data.append(item)
            self.size += 1
            return (1)
        return (0)

    def writeTexts(self):
        path = 'raw_text/'
        os.makedirs(path)
        for case in self.data:
            f = open(path + case.name + '_' + case.challenge + '_' + case.stage + '_' + case.labelled + '_' + case.label_type, 'w+')
            f.write(case.raw_text)
            f.close()
        print('raw_text Write Complete')

    def writeLabels(self):
        path = 'raw_labels/'
        os.makedirs(path)
        for case in self.data:
            if case.labelled == 'yes':
                f = open(path + case.name + '_' + case.challenge + '_' + case.stage + '_' + case.labelled + '_' + case.label_type, 'w+')
                f.write(case.raw_labels)
                f.close()
        print('raw_labels Write Complete')

    def numberOf(self, challenge=r'.', stage=r'.', labelled=r'.', label_type=r'.'):
        n = 0
        for case in self.data:
            if (re.match(challenge, case.challenge) != None) & \
                    (re.match(stage, case.stage) != None) & \
                    (re.match(labelled, case.labelled) != None) & \
                    (re.match(label_type, case.label_type) != None):
                n += 1
        return (n)

    def getDS(self, name=r'.', challenge=r'.', stage=r'.', labelled=r'.', label_type=r'.'):
        output = Set()
        for case in self.data:
            if (re.match(name, case.name) != None) & \
                    (re.match(challenge, case.challenge) != None) & \
                    (re.match(stage, case.stage) != None) & \
                    (re.match(labelled, case.labelled) != None) & \
                    (re.match(label_type, case.label_type) != None):
                output.add(case)
        return (output)

    def showInfo(self):
        for case in self.data:
            case.showInfo()
            print('\n')

    def append(self, dataset):
        self.data = self.data + dataset.data
        self.size += dataset.size

    def duplicates(self):
        dupl = 0
        for i in range(len(self.data)):
            occurence = 0
            for j in range(i, len(self.data)):
                if self.data[i].emb_text == self.data[j].emb_text:
                    occurence += 1
                    if occurence > 1:
                        occurence = 1
                        dupl += 1
        return (dupl)

    def exampleDuplicate(self):
        for i in range(len(self.data)):
            occurence = 0
            for j in range(i, len(self.data)):
                if self.data[i].emb_text == self.data[j].emb_text:
                    occurence += 1
                    if occurence > 1:
                        return (self.data[i], self.data[j])

    def addLabels(self, name, case, raw_labels):
        for i in range(self.size):
            if self.data[i].name == name:
                self.data[i].labelled = 'yes'
                if case == 'train':
                    self.data[i].label_type = 'train'
                if case == 'test':
                    self.data[i].label_type = 'test'
                self.data[i].raw_labels = raw_labels
                break

    def processForEmbedding(self):
        for i in range(self.size):
            self.data[i].processForEmbedding()

    def getSentences(self, challenge=r'.', stage=r'.'):
        sentences = []
        pool = self.getDS(challenge=challenge, stage=stage)
        for case in pool.data:
            for sent in case.emb_text:
                sentences.append(sent)
        return (sentences)