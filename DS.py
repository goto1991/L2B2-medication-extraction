import re


class DS:
    def __init__(self, name='', challenge='', stage='', raw_text=[]):
        self.name = name
        self.challenge = challenge
        self.stage = stage
        self.labelled = 'no'
        self.label_type = 'none'
        self.raw_text = raw_text
        self.emb_text = []
        self.token_text = []
        self.test_text = []
        self.raw_labels = []

    def process_for_embedding(self):
        self.emb_text = []
        temp = self.raw_text
        temp = re.sub(r'\d+', '<NUM>', temp)
        temp = re.sub(r'([A-Za-z]):', r'\1', temp)
        temp = re.sub(r'\n', ' ', temp)
        temp = re.sub(r'Dr\.', 'Dr', temp)
        temp = re.sub(r'Mr\.', 'Mr', temp)
        temp = re.sub(r'\. ([A-Z])', r'. A\1', temp)
        temp = re.sub(r'([A-Za-z]);', r'\1', temp)
        temp = re.split(r'\. [A-Z]', temp)
        for i in range(len(temp)):
            temp[i] = temp[i].lower()
            temp[i] = re.sub(r'([A-Za-z])\.', r'\1', temp[i])
            self.emb_text.append(temp[i].split())

    def process_for_testing(self):
        text = self.raw_text
        text = text.split('\n')
        for i in range(len(text)):
            text[i] = text[i].strip('.')  # Removing stops from end of lines
            text[i] = re.sub(r'\d+', '<NUM>', text[i])  # Substituting numbers with number tokens
            text[i] = re.sub(r'([A-Za-z]):', r'\1', text[i])  # Removing colons from letter words
            text[i] = re.sub(r'Dr\.', 'Dr', text[i])
            text[i] = re.sub(r'Mr\.', 'Mr', text[i])
            text[i] = re.sub(r'([A-Za-z])\.', r'\1', text[i])
            text[i] = re.sub(r'([A-Za-z]);', r'\1', text[i])
            text[i] = text[i].lower()
            text[i] = text[i].split()
        self.token_text = text

        indices = []
        second = False
        for term in re.finditer(r'm="[^|]+\|', self.raw_labels):
            term = term.group()
            index = []
            for window in re.finditer(r'\d+:\d+', term):
                index.append(list(map(int, window.group().split(':'))))
                if second:
                    if index not in indices:
                        indices.append(index)
                    index = []
                second = not second
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
        self.test_labels = truth
        self.test_text = [word for row in self.token_text for word in row]

    def show_info(self):
        print('Name: ', self.name)
        print('Challenge: ', self.challenge)
        print('Train or Test Set: ', self.stage)
        print('Labelled: ', self.labelled)
        print('Labeling Type: ', self.label_type)
