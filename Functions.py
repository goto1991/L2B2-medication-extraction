import os
import lxml.etree
import zipfile
import re
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
import random
import sklearn as sk
import colorama as col

from Set import pool
from DS import DS

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
# from bokeh.io import output_notebook
# £output_notebook()


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def first_time_load():
    dataset = pool()

    challenge = '2009 Medication Challenge'
    path = challenge + '/training.sets.released/'
    total = 0
    added = 0
    for folder in listdir_nohidden(path):
        for filename in listdir_nohidden(path + folder + '/'):
            total += 1
            with open(path + folder + '/' + filename, 'r') as file:
                temp = DS(name=filename, challenge=challenge, stage='train', raw_text=file.read())
                added += dataset.add(temp)
    print('%d/%d added from %s' % (added, total, path))
    path = challenge + '/test.released.8.17.09/'
    total = 0
    added = 0
    for filename in listdir_nohidden(path):
        total += 1
        with open(path + filename, 'r') as file:
            temp = DS(name=filename, challenge=challenge, stage='test', raw_text=file.read())
            added += dataset.add(temp)
    print('%d/%d added from %s' % (added, total, path))
    path = challenge + '/training.ground.truth/'
    for filename in listdir_nohidden(path):
        index = filename.split('_')[0]
        with open(path + filename, 'r') as file:
            dataset.add_labels(name=index, case='train', raw_labels=file.read())
    path = challenge + '/test.BYparticipant.ground_truth/converted.noduplicates.sorted/'
    for filename in listdir_nohidden(path):
        index = filename.split('.')[0]
        with open(path + filename, 'r') as file:
            dataset.add_labels(name=index, case='test', raw_labels=file.read())

    challenge = '2007 Smoking Challenge'
    path = [['', '/smokers_surrogate_train_all_version2.xml'], \
            ['', '/smokers_surrogate_test_all_groundtruth_version2.xml'], \
            ['/1C smokers_surrogate_train_all_version2.zip', 'smokers_surrogate_train_all_version2.xml'], \
            ['/1C smokers_surrogate_test_all_version2.zip', 'smokers_surrogate_test_all_version2.xml'], \
            ['/1C smokers_surrogate_test_all_groundtruth_version2.zip',
             'smokers_surrogate_test_all_groundtruth_version2.xml'], \
            ['/1B deid_surrogate_test_all_version2.zip', 'deid_surrogate_test_all_version2.xml'], \
            ['/1B deid_surrogate_train_all_version2_CORRECTED.zip', 'deid_surrogate_train_all_version2.xml'], \
            ['/1A unannotated_records_deid_smoking_CORRECTED.zip', 'unannotated_records_deid_smoking.xml']]
    for file in path:
        if file[0] == '':
            root = lxml.etree.parse(challenge + file[1]).getroot()
        else:
            with zipfile.ZipFile(challenge + file[0], 'r') as z:
                root = lxml.etree.parse(z.open(file[1])).getroot()
        names = []
        summaries = []
        for name in root.iter('RECORD'):
            names.append(name.attrib.get(name.attrib.keys()[0]))
        for summary in root.iter('TEXT'):
            summaries.append(summary.text)
        total = 0
        added = 0
        for i in range(len(names)):
            total += 1
            temp = DS(name=names[i], challenge=challenge, stage='train', raw_text=summaries[i])
            added += dataset.add(temp)
        print('%d/%d added from %s' % (added, total, file[0] + '/' + file[1]))

    challenge = '2008 Obesity Challenge'
    files = ['/obesity_patient_records_test.xml', \
             '/obesity_patient_records_training 50.xml', \
             '/obesity_patient_records_training.xml', \
             '/obesity_patient_records_training2.xml']
    for file in files:
        path = challenge + file
        root = lxml.etree.parse(path).getroot()
        names = []
        summaries = []
        for name in root.iter('doc'):
            names.append(name.attrib.get(name.attrib.keys()[0]))
        for summary in root.iter('text'):
            summaries.append(summary.text)
        total = 0
        added = 0
        for i in range(len(names)):
            total += 1
            temp = DS(name=names[i], challenge=challenge, stage='train', raw_text=summaries[i])
            added += dataset.add(temp)
        print('%d/%d added from %s' % (added, total, path))

    challenge = '2010 Relations Challenge'
    folders = ['/concept_assertion_relation_training_data/beth/txt/', \
               '/concept_assertion_relation_training_data/partners/txt/', \
               '/concept_assertion_relation_training_data/partners/unannotated/', \
               '/Sample Data/', \
               '/test_data/']
    for folder in folders:
        path = challenge + folder
        total = 0
        added = 0
        for filename in listdir_nohidden(path):
            if filename.endswith('.txt'):
                total += 1
                with open(path + filename, 'r') as file:
                    temp = DS(name=filename, challenge=challenge, stage='test', raw_text=file.read())
                    added += dataset.add(temp)
        print('%d/%d added from %s' % (added, total, path))

    challenge = '2011 Coreference Challenge'
    folders = ['/Test_Beth/docs/', \
               '/Test_Partners/docs/', \
               '/Train_Beth/docs/', \
               '/Train_Partners/docs/']
    for folder in folders:
        path = challenge + folder
        total = 0
        added = 0
        for filename in listdir_nohidden(path):
            if filename.endswith('.txt'):
                total += 1
                with open(path + filename, 'r') as file:
                    temp = DS(name=filename, challenge=challenge, stage='test', raw_text=file.read())
                    added += dataset.add(temp)
        print('%d/%d added from %s' % (added, total, path))

    challenge = '2012 Temporal Relations Challenge'
    folders = ['/Training Full_2012-07-15.original-annotation.release/', \
               '/Evaluation_Test TIMEX groundtruth_2012-08-08.test-data.event-timex-groundtruth/i2b2/', \
               '/Evaluation_Test ground_truth_2012-08-23.test-data.groundtruth/merged_i2b2/', \
               '/Evaluation_Test ground_truth_2012-08-23.test-data.groundtruth/unmerged_i2b2/', \
               '/Evaluation_Test data_2012-08-06.test-data-release/txt/']
    for folder in folders:
        path = challenge + folder
        total = 0
        added = 0
        for filename in listdir_nohidden(path):
            if filename.endswith('.txt'):
                total += 1
                with open(path + filename, 'r') as file:
                    temp = DS(name=filename, challenge=challenge, stage='test', raw_text=file.read())
                    added += dataset.add(temp)
        print('%d/%d added from %s' % (added, total, path))

    return(dataset)


def label_words(Dataset, model):
    doc = open("stopwords.txt", "r")
    stopwords = set(doc.read().split('\n'))
    doc.close()
    stopwords.update({'nm', 'ngl'})

    medications = set()
    dosages = set()
    modes = set()
    frequencies = set()
    durations = set()
    reasons = set()

    vocab = model.wv.vocab.keys()
    labelled = Dataset.get_DS(labelled='yes')

    for case in labelled.data:
        for term in re.finditer(r'm="[^"]+"', case.raw_labels):
            term = term.group()[3:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [medications.add(word) for word in term.split() if (word not in stopwords) and word in vocab]
        for term in re.finditer(r'do="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [dosages.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'mo="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [modes.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'f="[^"]+"', case.raw_labels):
            term = term.group()[3:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [frequencies.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'du="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [durations.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'r="[^"]+"', case.raw_labels):
            term = term.group()[3:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'([A-Za-z])\.', r'\1', term)
            term = re.sub(r'([A-Za-z]);', r'\1', term)
            term = re.sub(r'([A-Za-z]):', r'\1', term)
            term = re.sub(r'[()]', ' ', term)
            [reasons.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]

    print('Number of: m={}, do={}, mo={}, f={}, du={}, r={}'.format(len(medications), len(dosages), len(modes), len(frequencies), len(durations), len(reasons)))
    return medications, dosages, modes, frequencies, durations, reasons


def write_sentences(sentences, path):
    os.makedirs(path)
    f = open(path + '/sentences', 'w+')
    for sent in sentences:
        for word in sent:
            f.write(word + ' ')
        f.write('\n')
    f.close()
    print('Sentence Write Complete')


def load_sentences(path):
    sentences = []
    f = open(path + '/sentences', 'r')
    for line in f:
        sentences.append(line.split())
    f.close()
    print('Sentence Load Complete')
    return sentences


def write_emb(title, model):
    try:
        os.remove(title)
    except OSError:
        pass
    f = open(title, 'w+')
    f.write(' '.join([str(len(model.wv.vocab)), str(model.vector_size)]) + '\n')
    for word in model.wv.vocab:
        f.write(word + '\n')
        f.write(' '.join(str(feature) for feature in model[word]) + '\n')
    f.close()
    print('Embeddings of %d words with length %d have ben written to %s' % (len(model.wv.vocab), model.vector_size, title))


def load_emb(title):
    model = {}
    f = open(title, 'r')
    word_num, vec_size = list(map(int, f.readline().split()))
    for i in range(word_num):
        word = f.readline().strip()
        model[word] = list(map(float, f.readline().split()))
    f.close()
    print('Embeddings of %d words with length %d loaded from %s' % (word_num, vec_size, title))
    return model


def write_labels(dict, path):
    os.makedirs(path)
    for target in dict.keys():
        temp = sorted((dict[target]))
        f = open(path + '/' + target, 'w+')
        for word in temp:
            f.write(word + '\n')
        f.close()
    print('Label Write Complete')


def load_labels(path):
    dict = {}
    for file in os.listdir(path):
        dict[file] = set()
        f = open(path + '/' + file)
        for line in f:
            dict[file].add(line.strip())
        f.close()
    print('Label Load Complete')
    return dict


def write_word_indices(word_indices, path):
    os.makedirs(path)
    with open(path + '/word_indices', 'w+') as f:
        for word in word_indices:
            f.write(word + ' ' + str(word_indices[word]))
            f.write('\n')
    print('Word Indices Write Complete')


def load_word_indices(path):
    word_indices = {}
    with open(path + '/word_indices', 'r') as f:
        for line in f:
            word, index = line.split()
            word_indices[word] = int(index)
    print('Word Indices Load Complete')
    return word_indices


def write_emb_layer(emb_layer, path):
    os.makedirs(path)
    with open(path + '/emb_layer', 'w+') as f:
        word_num, emb_size = np.shape(emb_layer)
        f.write(' '.join([str(word_num), str(emb_size)]) + '\n')
        for embedding in emb_layer:
            f.write(' '.join(str(feature) for feature in embedding) + '\n')
    print('Embedding Layer Write Complete')


def load_emb_layer(path):
    with open(path + '/emb_layer', 'r') as f:
        word_num, emb_size = list(map(int, f.readline().split()))
        emb_layer = np.empty([word_num, emb_size], dtype=np.float32)
        for i in range(word_num):
            emb_layer[i] = np.array((list(map(np.float32, f.readline().split()))))
    print('Embedding Layer Load Complete')
    return emb_layer


def visualise(model, sentences, labels, topn=1000, title='T-SNE'):
    words = [word for sent in sentences for word in sent]
    cnt = np.array(Counter(words).most_common(topn))
    top_words = np.ndarray.tolist(cnt[:, 0])

    visualisation = []
    colormap = []
    colors = ['red', 'green',  'yellow', 'purple', 'orange', 'cyan', 'blue']

    [(visualisation.append(word), colormap.append(colors[i])) for i in range(len(labels)) for word in labels[i]]
    [(visualisation.append(word), colormap.append(colors[6])) for word in top_words if word not in visualisation]
    words_vec = model[visualisation]

    tsne = TSNE(n_components=2, random_state=0)
    words_tsne = tsne.fit_transform(words_vec)

    p = figure(tools="pan,wheel_zoom,reset,save",
               toolbar_location="above",
               title=title,
               plot_width = 1000,
               plot_height = 1000)

    source = ColumnDataSource(data=dict(x1=words_tsne[:, 0],
                                        x2=words_tsne[:, 1],
                                        coloring=colormap))

    p.scatter(x="x1", y="x2", color="coloring", size=8, source=source)

    #labels = LabelSet(x="x1", y="x2", y_offset=6, source=source)
    #p.add_layout(labels)

    show(p)


def generate_naive_traintest(vocab, labels, train_size, test_size, train_label_percentage=10, test_label_percentage=10, word_repetition=True, label_repetition=True, report=False):
    train_set = []
    train_labels = []
    train_size = train_size
    test_set = []
    test_labels = []
    test_size = test_size

    vocab = set(vocab)
    target = set(labels)

    for i in range(train_size * (100 - train_label_percentage) // 100):
        word = random.sample(vocab, 1)
        while word[0] in labels:
            word = random.sample(vocab, 1)
        train_set.append(word[0])
        train_labels.append([0, 1])
        if not word_repetition: vocab.discard(word[0])

    for i in range(train_size * train_label_percentage // 100):
        word = random.sample(target, 1)
        train_set.append(word[0])
        train_labels.append([1, 0])
        if not label_repetition: target.discard(word[0])

    for i in range(test_size * (100 - test_label_percentage) // 100):
        word = random.sample(vocab, 1)
        while word[0] in labels:
            word = random.sample(vocab, 1)
        test_set.append(word[0])
        test_labels.append([0, 1])
        if not word_repetition: vocab.discard(word[0])

    for i in range(test_size * test_label_percentage // 100):
        word = random.sample(target, 1)
        test_set.append(word[0])
        test_labels.append([1, 0])
        if not label_repetition: target.discard(word[0])

    if report:
        print('Train set size: %d \tPercentage of labels: %f' %(len(train_labels), np.array(train_labels).sum(0)[0]/len(train_labels)))
        print('Test set size: %d \tPercentage of labels: %f' %(len(test_labels), np.array(test_labels).sum(0)[0] / len(test_labels)))

    word_sets = {'train_set': train_set, 'train_labels': train_labels, 'test_set': test_set, 'test_labels': test_labels}
    return word_sets


def embed_words(word_sets, model):
    emb_sets = {}
    emb_sets['train_set'] = [model[word] for word in word_sets['train_set']]
    emb_sets['train_labels'] = word_sets['train_labels']
    emb_sets['test_set'] = [model[word] for word in word_sets['test_set']]
    emb_sets['test_labels'] = word_sets['test_labels']
    return emb_sets


def saturate_training_set(sets, share, seqlen=False):
    targets = []
    for i in range(len(sets['train_set'])):
        if sets['train_labels'][i] == [1, 0]: targets.append(sets['train_set'][i])

    if seqlen:
        lengths = []
        for i in range(len(sets['train_set'])):
            if sets['train_labels'][i] == [1, 0]: lengths.append(sets['train_lengths'][i])

    while (np.array(sets['train_labels']).sum(0) / len(sets['train_labels']))[0] < share:
        for i in range(len(targets)):
            sets['train_set'].append(targets[i])
            sets['train_labels'].append([1, 0])
            if seqlen:
                sets['train_lengths'].append(lengths[i])


def get_index_and_emb_layer(model):
    word_indices = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<go>': 3, '<eos>': 4}
    emb_layer = np.empty([len(model.wv.vocab) + 5, model.vector_size])
    emb_layer[:5] = np.zeros(model.vector_size)
    i = 5

    for word in model.wv.vocab.keys():
        word_indices[word] = i
        emb_layer[i] = model[word]
        i += 1
    return word_indices, emb_layer


def token_perf(res, tru, tfpn=True, precision = True, recall=True,  f1=True):
    perf_dict = {}
    perf_dict['tp'] = len([1 for a in range(len(tru)) if (res[a] == 0) and (tru[a] == 0)])
    perf_dict['tn'] = len([1 for a in range(len(tru)) if (res[a] == 1) and (tru[a] == 1)])
    perf_dict['fp'] = len([1 for a in range(len(tru)) if (res[a] == 0) and (tru[a] == 1)])
    perf_dict['fn'] = len([1 for a in range(len(tru)) if (res[a] == 1) and (tru[a] == 0)])
    perf_dict['precision'] = perf_dict['tp'] / (perf_dict['tp'] + perf_dict['fp'])
    perf_dict['recall'] = perf_dict['tp'] / (perf_dict['tp'] + perf_dict['fn'])
    perf_dict['f1'] = sk.metrics.f1_score(tru, res, pos_label=0, average='binary')
    if tfpn:
        print('TP\tTN\tFP\tFN\n{}\t{}\t{}\t{}\n'.format(perf_dict['tp'], perf_dict['tn'], perf_dict['fp'], perf_dict['fn']))
    if precision:
        print('Precision: {:.4f}'.format(perf_dict['precision']))
    if recall:
        print('Recall: {:.4f}'.format(perf_dict['recall']))
    if f1:
        print('F1-Score: {:.4f}'.format(perf_dict['f1']))
    return perf_dict


def phrase_perf(target, NN, testers, model, side_words=[0, 0], tfpn=True, precision=True, recall=True, f1=True, case_info=False, show_phrases=False, rnn=False):
    perf_dict = {'tp': 0, 'fp': 0, 'fn': 0}

    for i in range(testers.size):
        current_ds = pool(data=[testers.data[i]])
        if case_info:
            current_ds.show_info()
        current_ds.process_for_testing(target)

        if not rnn:
            cur_set = current_ds.get_ff_sets(model=model, left_words=side_words[0], right_words=side_words[1])
            cur_res = NN.predict(cur_set[0])
        else:
            cur_set = current_ds.get_rnn_sets(word_indices=model, left_words=side_words[0], right_words=side_words[1])
            cur_res = NN.predict(cur_set[0], cur_set[3])

        rowed_res = []
        cursor = 0
        for row in current_ds.data[0].token_text:
            rowed_res.append(cur_res[cursor:cursor + len(row)])
            cursor += len(row)

        res_offsets = []
        res_phrases = []
        res_inside = False
        for i in range(len(current_ds.data[0].token_text)):
            for j in range(len(current_ds.data[0].token_text[i])):
                if not res_inside:
                    if rowed_res[i][j] == 0:
                        cur_phrase = current_ds.data[0].token_text[i][j]
                        cur_offset = [[i + 1, j], [i + 1, j]]
                        res_inside = True
                else:
                    if rowed_res[i][j] == 0:
                        cur_phrase = cur_phrase + ' ' + current_ds.data[0].token_text[i][j]
                        cur_offset[1] = [i + 1, j]
                    else:
                        res_phrases.append(cur_phrase)
                        res_offsets.append(cur_offset)
                        res_inside = False

        tru_fields = []
        tru_phrases = []
        tru_offsets = []
        for term in re.finditer(re.escape(target) + r'="[^|]+\|', current_ds.data[0].raw_labels):
            term = term.group()
            if not re.search(r'="nm"', term):
                if term not in tru_fields:
                    tru_fields.append(term)
                    tru_phrases.append(re.match(re.escape(target) + r'="[^"]+"', term).group()[len(target) + 2:-1])
                    cur_offset = []
                    for window in re.finditer(r'\d+:\d+', term):
                        cur_offset.append(list(map(int, window.group().split(':'))))
                    tru_offsets.append(cur_offset)


        for i in range(len(res_phrases)):
            if res_offsets[i] in tru_offsets:
                j = tru_offsets.index(res_offsets[i])
                perf_dict['tp'] += 1
                if show_phrases:
                    print(res_phrases[i], *res_offsets[i], tru_phrases[j], *tru_offsets[j])
            else:
                perf_dict['fp'] += 1
                if show_phrases:
                    print(col.Back.YELLOW, end='')
                    print(res_phrases[i], *res_offsets[i])
                    print(col.Back.RESET, end='')

        for i in range(len(tru_phrases)):
            if tru_offsets[i] not in res_offsets:
                perf_dict['fn'] += 1
                if show_phrases:
                    print(col.Back.RED, end='')
                    print(tru_phrases[i], *tru_offsets[i])
                    print(col.Back.RESET, end='')
        if show_phrases:
            print('\n')

    perf_dict['precision'] = perf_dict['tp'] / (perf_dict['tp'] + perf_dict['fp'])
    perf_dict['recall'] = perf_dict['tp'] / (perf_dict['tp'] + perf_dict['fn'])
    perf_dict['f1'] = (2 * perf_dict['tp']) / (2 * perf_dict['tp'] + perf_dict['fn'] + perf_dict['fp'])

    if tfpn:
        print('TP\tFP\tFN\n{}\t{}\t{}\n'.format(perf_dict['tp'], perf_dict['fp'], perf_dict['fn']))
    if precision:
        print('Precision: {:.4f}'.format(perf_dict['precision']))
    if recall:
        print('Recall: {:.4f}'.format(perf_dict['recall']))
    if f1:
        print('F1-Score: {:.4f}'.format(perf_dict['f1']))
    return perf_dict


def category_words(words, res, tru, res_label, tru_label):
    print(', '.join([words[a] for a in range(len(words)) if res[a] == res_label and tru[a] == tru_label]))


def colour_text(words, res, tru, segment=None):
    out = []
    if segment == None: segment = [0, len(words)]
    for a in range(len(words[segment[0]:segment[1]])):
        if res[segment[0] + a] == 0 and tru[segment[0] + a] == 0:
            out.append(col.Back.GREEN + words[segment[0] + a].upper() + col.Back.RESET)
        elif res[segment[0] + a] == 0 and tru[segment[0] + a] == 1:
            out.append(col.Back.YELLOW + words[segment[0] + a].upper() + col.Back.RESET)
        elif res[segment[0] + a] == 1 and tru[segment[0] + a] == 0:
            out.append(col.Back.RED + words[segment[0] + a].upper() + col.Back.RESET)
        else:
            out.append(words[segment[0] + a])

    print(' '.join(out))

