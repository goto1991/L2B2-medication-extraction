import re
import os

import Functions as fn
from DS import DS


class pool:
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
        for i in range(self.size):
            f = open(path + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
            f.write(self.data[i].raw_text)
            f.close()
        print('raw_text Write Complete')

    def writeLabels(self):
        path = 'raw_labels/'
        os.makedirs(path)
        for i in range(self.size):
            if self.data[i].labelled == 'yes':
                f = open(path + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
                f.write(self.data[i].raw_labels)
                f.close()
        print('raw_labels Write Complete')

    def loadTexts(self):
        for filename in fn.listdir_nohidden('raw_text'):
            info = filename.split('_')
            with open('raw_text/' + filename, 'r') as file:
                temp = DS(name=info[0], challenge=info[1], stage=info[2], raw_text=file.read())
                self.add(temp)
        print('Texts loaded')

    def loadLabels(self):
        for filename in fn.listdir_nohidden('raw_labels'):
            info = filename.split('_')
            with open('raw_labels/' + filename, 'r') as file:
                self.addLabels(name=info[0], case=info[4], raw_labels=file.read())
        print('Labels loaded')

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
        output = pool()
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
        for case in dataset:
            self.add(case)

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
                self.data[i].label_type = case
                self.data[i].raw_labels = raw_labels
                break

    def processForEmbedding(self):
        for i in range(self.size):
            self.data[i].processForEmbedding()

    def getSentences(self, challenge=r'.', stage=r'.'):
        sentences = []
        temp = self.getDS(challenge=challenge, stage=stage)
        for case in temp.data:
            for sent in case.emb_text:
                sentences.append(sent)
        return (sentences)