import os
import lxml.etree
import zipfile
import re
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
import random

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
            term = re.sub(r'[()]', ' ', term)
            [medications.add(word) for word in term.split() if (word not in stopwords) and word in vocab]
        for term in re.finditer(r'do="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'[()]', ' ', term)
            [dosages.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'mo="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'[()]', ' ', term)
            [modes.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'f="[^"]+"', case.raw_labels):
            term = term.group()[3:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'[()]', ' ', term)
            [frequencies.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'du="[^"]+"', case.raw_labels):
            term = term.group()[4:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
            term = re.sub(r'[()]', ' ', term)
            [durations.add(word) for word in term.split() if (word not in stopwords) and (word in vocab)]
        for term in re.finditer(r'r="[^"]+"', case.raw_labels):
            term = term.group()[3:-1]
            term = re.sub(r'\d+', '<num>', term)
            term = re.sub(r'\.\.\.', ' ', term)
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
                                        names=visualisation,
                                        coloring=colormap))

    p.scatter(x="x1", y="x2", color="coloring", size=8, source=source)

    labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                      text_font_size="8pt", text_color="#555555",
                      source=source, text_align='center')
    p.add_layout(labels)

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


def get_ff_traintest(dataset, model, left_words=0, right_words=0):
    padded_texts = [['<pad>' for i in range(left_words)] + case.test_text + ['<pad>' for i in range(right_words)] for case in dataset.data]

    vectorized_texts = []
    for text in padded_texts:
        text_vector = []
        for i in range(left_words, len(text) - right_words):
            word_vector = [text[i + pos] for pos in range(-left_words, right_words + 1)]
            word_vector = np.concatenate(
                [model[word] if word in model.wv.vocab else np.zeros(model.vector_size) for word in word_vector])
            text_vector.append(word_vector)
        vectorized_texts.append(text_vector)

    ten_percent = len(dataset.data) // 10

    train_set = [vector for case in vectorized_texts[:-2 * ten_percent] for vector in case]
    train_labels = [label for case in dataset.data[:-2 * ten_percent] for label in case.test_labels]
    validation_set = [vector for case in vectorized_texts[-2 * ten_percent:-ten_percent] for vector in case]
    validation_labels = [label for case in dataset.data[-2 * ten_percent:-ten_percent] for label in case.test_labels]
    test_set = [vector for case in vectorized_texts[-ten_percent:] for vector in case]
    test_labels = [label for case in dataset.data[-ten_percent:] for label in case.test_labels]
    test_words = [word for case in dataset.data[-ten_percent:] for word in case.test_text]

    return {'train_set': train_set,
            'train_labels': train_labels,
            'validation_set': validation_set,
            'validation_labels': validation_labels,
            'test_set': test_set, 'test_labels': test_labels,
            'test_words': test_words}


def saturate_training_set_labels(dataset, model, labels, share):
    while (np.array(dataset['train_labels']).sum(0) / len(dataset['train_labels']))[0] < share:
        for med in labels:
            dataset['train_set'].append(model[med])
            dataset['train_labels'].append([1, 0])
        print('Label proportion: %f' % (np.array(dataset['train_labels']).sum(0) / len(dataset['train_labels']))[0], end='\r')


def saturate_training_set_training(dataset, share):
    targets = []
    for i in range(len(dataset['train_set'])):
        if dataset['train_labels'][i] == [1, 0]: targets.append(dataset['train_set'][i])

    while (np.array(dataset['train_labels']).sum(0) / len(dataset['train_labels']))[0] < share:
        for med in targets:
            dataset['train_set'].append(med)
            dataset['train_labels'].append([1, 0])
        print('Label proportion: {:.3f}'.format(
            (np.array(dataset['train_labels']).sum(0) / len(dataset['train_labels']))[0]), end='\r')
