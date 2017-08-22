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
        self.enc_labels = []
        self.enc_inputs = []
        self.dec_inputs = []
        self.dec_outputs = []

    def process_for_embedding(self):
        self.emb_text = []
        temp = self.raw_text
        temp = re.sub(r'\d+', '<NUM>', temp)
        temp = re.sub(r'([A-Za-z]):', r'\1', temp)
        temp = re.sub(r'\n', ' ', temp)
        temp = re.sub(r'Dr\.', 'Dr', temp)
        temp = re.sub(r'Mr\.', 'Mr', temp)
        temp = re.sub(r'([A-Za-z]);', r'\1', temp)
        temp = re.sub(r'\. ([A-Z])', r'. A\1', temp)
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
            text[i] = re.sub(r'([A-Za-z]);', r'\1', text[i])
            text[i] = re.sub(r'([A-Za-z])\.', r'\1', text[i])
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

        label_text = [[0 for word in row] for row in self.token_text]

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
                        for offset in windows:
                            tar_ind[i][index].append(offset)

        label_text = [[[0] for word in row] for row in self.token_text]
        for i in range(len(tar_ind)):
            for entry in range(entry_num):
                index = 0
                while index < len(tar_ind[i][entry]):
                    start_row = tar_ind[i][entry][index][0] - 1
                    start_word = tar_ind[i][entry][index][1]
                    end_row = tar_ind[i][entry][index + 1][0] - 1
                    end_word = tar_ind[i][entry][index + 1][1]
                    row = start_row
                    word = start_word
                    while (row <= end_row) and ((row < end_row) or (word <= end_word)):
                        row_len = len(label_text[row])
                        label_text[row][word] = [i+1]
                        if word < row_len - 1:
                            word += 1
                        else:
                            word = 0
                            row += 1
                    index += 2

        all_enc_labels = []
        all_enc_inputs = []
        all_dec_inputs = []
        all_dec_outputs = []

        for entry in range(entry_num):
            current_enc_input = []
            current_dec_input = ['<go>']
            min_row = float('inf')
            max_row = 0
            index = 0
            med_indices = []
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
                    current_enc_input.append(self.token_text[row][word])
                    med_indices.append([row, word])
                    if word < row_len - 1:
                        word += 1
                    else:
                        word = 0
                        row += 1
                index += 2
            current_enc_input.append('<start>')

            start = max(0, min_row - 2)
            end = min(row_num, max_row + 3)
            dummy_dec_input = []
            for i in range(start, end):
                dummy_row = []
                for word in self.token_text[i]:
                    current_enc_input.append(word)
                    dummy_row.append(0)
                dummy_dec_input.append(dummy_row)

            for index in med_indices:
                dummy_dec_input[index[0]-start][index[1]] = 6

            for i in range(1, 6):
                index = 0
                while index < len(tar_ind[i][entry]):
                    if tar_ind[i][entry][index][0] != -1:
                        start_row = tar_ind[i][entry][index][0] - 1 - start
                        start_word = tar_ind[i][entry][index][1]
                        end_row = tar_ind[i][entry][index + 1][0] - 1 - start
                        end_word = tar_ind[i][entry][index + 1][1]
                        if end_row > len(dummy_dec_input) - 1:
                            end_row = len(dummy_dec_input) - 1
                            end_word = len(dummy_dec_input[end_row]) - 1
                        if start_row < 0:
                            start_row = 0
                            start_word = 0
                        row = start_row
                        word = start_word
                        # print(i, start_row, start_word, end_row, end_word)
                        while (row <= end_row) and ((row < end_row) or (word <= end_word)):
                            row_len = len(dummy_dec_input[row])
                            dummy_dec_input[row][word] = i
                            if word < row_len - 1:
                                word += 1
                            else:
                                word = 0
                                row += 1
                    index += 2

            for row in dummy_dec_input:
                for word in row:
                    current_dec_input.append(word)

            current_dec_output = current_dec_input[1:]
            current_dec_output.append(8)

            start_tok = current_enc_input.index('<start>')
            new_dec_input=['<go>']
            for i in range(1, len(current_dec_input)):
                if current_dec_input[i] is not 0:
                    new_dec_input.append(current_enc_input[start_tok + i])
            new_dec_output = new_dec_input[1:]
            new_dec_output.append('<eos>')

            #start = max(0, min_row - 2)
            #end = min(row_num, max_row + 3)
            window_labels = [[1] for i in range(start_tok)]
            window_labels.append([0])
            for i in range(start, end):
                for label in label_text[i]:
                    window_labels.append(label)

            all_enc_labels.append(window_labels)
            all_enc_inputs.append(current_enc_input)
            all_dec_inputs.append(new_dec_input)
            all_dec_outputs.append(new_dec_output)

        self.enc_labels = all_enc_labels
        self.enc_inputs = all_enc_inputs
        self.dec_inputs = all_dec_inputs
        self.dec_outputs = all_dec_outputs

    def show_info(self):
        print('Name: ', self.name)
        print('Challenge: ', self.challenge)
        print('Train or Test Set: ', self.stage)
        print('Labelled: ', self.labelled)
        print('Labeling Type: ', self.label_type)
