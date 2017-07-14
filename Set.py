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
                return True
        return False

    def add(self, item):
        if not self.present(item):
            self.data.append(item)
            self.size += 1
            return 1
        return 0

    def write_texts(self):
        path = 'raw_text/'
        os.makedirs(path)
        for i in range(self.size):
            f = open(path + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
            f.write(self.data[i].raw_text)
            f.close()
        print('raw_text Write Complete')

    def write_labels(self):
        path = 'raw_labels/'
        os.makedirs(path)
        for i in range(self.size):
            if self.data[i].labelled == 'yes':
                f = open(path + str(i).zfill(4) + '_' + self.data[i].challenge + '_' + self.data[i].stage + '_' + self.data[i].labelled + '_' + self.data[i].label_type + '_' + self.data[i].name, 'w+')
                f.write(self.data[i].raw_labels)
                f.close()
        print('raw_labels Write Complete')

    def load_texts(self):
        for filename in fn.listdir_nohidden('raw_text'):
            info = filename.split('_')
            with open('raw_text/' + filename, 'r') as file:
                temp = DS(name=info[0], challenge=info[1], stage=info[2], raw_text=file.read())
                self.add(temp)
        print('Texts loaded')

    def load_labels(self):
        for filename in fn.listdir_nohidden('raw_labels'):
            info = filename.split('_')
            with open('raw_labels/' + filename, 'r') as file:
                self.add_labels(name=info[0], case=info[4], raw_labels=file.read())
        print('Labels loaded')

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
        output = pool()
        for case in self.data:
            if (re.match(name, case.name) != None) & \
                    (re.match(challenge, case.challenge) != None) & \
                    (re.match(stage, case.stage) != None) & \
                    (re.match(labelled, case.labelled) != None) & \
                    (re.match(label_type, case.label_type) != None):
                output.add(case)
        return (output)

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

    def add_labels(self, name, case, raw_labels):
        for i in range(self.size):
            if self.data[i].name == name:
                self.data[i].labelled = 'yes'
                self.data[i].label_type = case
                self.data[i].raw_labels = raw_labels
                break

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
        for case in self.data:
            text = case.raw_text
            text = text.split('\n')
            for i in range(len(text)):
                text[i] = text[i].strip('.')                    # Removing stops from end of lines
                text[i] = re.sub(r'\d+', '<NUM>', text[i])      # Substituting numbers with number tokens
                text[i] = re.sub(r'([A-Za-z]):', r'\1', text[i])# Removing colons from letter words
                text[i] = re.sub(r'Dr.', 'Dr', text[i])
                text[i] = re.sub(r'Mr.', 'Mr', text[i])
                text[i] = re.sub(r'\. ([A-Z])', r'. A\1', text[i]) # Adding capial letter after new sentence
                text[i] = re.sub(r'\. [A-Z]', ' ', text[i]) # Removing end of sentence stops
                text[i] = text[i].lower()
                text[i] = text[i].split()
            case.token_text = text

            indices = []
            for med in re.finditer(r'm="[^"]*" \d+:\d+ \d+:\d+', case.raw_labels):
                indices.append([[int(a) for a in b.split(':')] for b in med.group().split()[-2:]])
            indices.sort()
            indices.append([[0, 0], [0, 0]])

            truth = []
            c = 0
            inside = False
            for i in range(len(text)):
                for j in range(len(text[i])):
                    if inside:
                        if i + 1 < indices[c][1][0]:
                            truth.append([1, 0])
                        elif i + 1 == indices[c][1][0]:
                            if j < indices[c][1][1]:
                                truth.append([1, 0])
                            elif j == indices[c][1][1]:
                                truth.append([1, 0])
                                inside = False
                                c += 1
                    else:
                        if [i + 1, j] == indices[c][0]:
                            truth.append([1, 0])
                            if [i + 1, j] == indices[c][1]:
                                c += 1
                            else:
                                inside = True
                        else:
                            truth.append([0, 1])
            case.test_labels = truth
            case.test_text = [word for sentence in case.token_text for word in sentence]