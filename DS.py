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
        self.test_labels = []

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

    def process_for_testing(self, target):
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
        for term in re.finditer(re.escape(target) + r'="[^|]+\|', self.raw_labels):
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

    def process_for_s2s_testing(self):
        #self.show_info()
        text = self.raw_text
        text = text.split('\n')
        row_num = len(text)
        for i in range(row_num):
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

        tar_ind = [[], [], [], [], [], []]
        entry_num = 0
        for label_line in self.raw_labels.split('\n'):
            # print(len(tar_ind[0]))
            fields = label_line.split('||')
            if len(fields) > 5:
                windows = []
                for window in re.finditer(r'\d+:\d+', fields[0]):
                    windows.append(list(map(int, window.group().split(':'))))

                if windows not in tar_ind[0]:
                    tar_ind[0].append(windows)
                    index = -1
                    entry_num += 1
                else:
                    index = tar_ind[0].index(windows)

                for i in range(1, 6):
                    windows = []
                    if not re.search(r'="nm"', fields[i]):
                        for window in re.finditer(r'\d+:\d+', fields[i]):
                            windows.append(list(map(int, window.group().split(':'))))
                    else:
                        windows = [[-1, -1], [-1, -1]]
                    if index == -1:
                        tar_ind[i].append(windows)
                    else:
                        for num in windows:
                            tar_ind[i][index].append(num)

        all_texts = []
        all_labels = []

        for entry in range(entry_num):
            current_text = []
            current_labels = [6]
            min_row = float('inf')
            max_row = 0
            index = 0
            while index < len(tar_ind[0][entry]):
                start_row = tar_ind[0][entry][index][0] - 1
                start_word = tar_ind[0][entry][index][1]
                end_row = tar_ind[0][entry][index + 1][0] - 1
                end_word = tar_ind[0][entry][index + 1][1]
                min_row = min(min_row, start_row)
                max_row = max(max_row, end_row)
                row = start_row
                word = start_word
                # print(start_row, start_word, end_row, end_word)
                while (row <= end_row) and ((row < end_row) or (word <= end_word)):
                    row_len = len(self.token_text[row])
                    current_text.append(self.token_text[row][word])
                    current_labels.append(0)
                    if word < row_len - 1:
                        word += 1
                    else:
                        word = 0
                        row += 1
                current_text.append('<pad>')
                current_labels.append(0)
                index += 2

            start = max(0, min_row - 2)
            end = min(row_num, max_row + 3)
            dummy_labels = []
            for i in range(start, end):
                dummy = []
                for word in self.token_text[i]:
                    current_text.append(word)
                    dummy.append(0)
                dummy_labels.append(dummy)

            for i in range(1, 6):
                index = 0
                while index < len(tar_ind[i][entry]):
                    if tar_ind[i][entry][index][0] != -1:
                        start_row = tar_ind[i][entry][index][0] - 1 - start
                        start_word = tar_ind[i][entry][index][1]
                        end_row = tar_ind[i][entry][index + 1][0] - 1 - start
                        end_word = tar_ind[i][entry][index + 1][1]
                        row = start_row
                        word = start_word
                        if end_row > len(dummy_labels) - 1:
                            end_row = len(dummy_labels) - 1
                            end_word = len(dummy_labels[end_row]) - 1
                        # print(i, start_row, start_word, end_row, end_word)
                        while (row <= end_row) and ((row < end_row) or (word <= end_word)):
                            row_len = len(dummy_labels[row])
                            dummy_labels[row][word] = i
                            if word < row_len - 1:
                                word += 1
                            else:
                                word = 0
                                row += 1
                    index += 2

            for row in dummy_labels:
                for word in row:
                    current_labels.append(word)

            all_texts.append(current_text)
            all_labels.append(current_labels)

        self.test_text = all_texts
        self.test_labels = all_labels

    def show_info(self):
        print('Name: ', self.name)
        print('Challenge: ', self.challenge)
        print('Train or Test Set: ', self.stage)
        print('Labelled: ', self.labelled)
        print('Labeling Type: ', self.label_type)
