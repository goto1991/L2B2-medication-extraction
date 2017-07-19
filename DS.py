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
        self.raw_labels = []

    def process_for_embedding(self):
        self.emb_text = []
        temp = self.raw_text
        temp = re.sub(r'\d+', '<NUM>', temp)
        temp = re.sub(r'([A-Za-z]):', r'\1', temp)
        temp = re.sub(r'\n', ' ', temp)
        temp = re.sub(r'Dr.', 'Dr', temp)
        temp = re.sub(r'Mr.', 'Mr', temp)
        temp = re.sub(r'\. ([A-Z])', r'. A\1', temp)
        temp = re.split(r'\. [A-Z]', temp)
        for i in range(len(temp)):
            temp[i] = temp[i].lower()
            self.emb_text.append(temp[i].split())

    def show_info(self):
        print('Name: ', self.name)
        print('Challenge: ', self.challenge)
        print('Train or Test Set: ', self.stage)
        print('Labelled: ', self.labelled)
        print('Labeling Type: ', self.label_type)
