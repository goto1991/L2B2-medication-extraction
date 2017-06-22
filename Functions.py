import os
import lxml.etree
import zipfile
import re
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()

from Set import pool
from DS import DS


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
            dataset.addLabels(name=index, case='train', raw_labels=file.read())
    path = challenge + '/test.BYparticipant.ground_truth/converted.noduplicates.sorted/'
    for filename in listdir_nohidden(path):
        index = filename.split('.')[0]
        with open(path + filename, 'r') as file:
            dataset.addLabels(name=index, case='test', raw_labels=file.read())

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

def label_words(Dataset):
    doc = open("stopwords.txt", "r")
    stopwords = set(doc.read().split('\n'))
    stopwords.update({'nm', 'ngl'})

    medications = set()
    dosages = set()
    modes = set()
    frequencies = set()
    durations = set()
    reasons = set()

    vocab = {word for sent in Dataset.getSentences() for word in sent}

    labelled = Dataset.getDS(labelled='yes')

    for case in labelled.data:
        for term in re.finditer(r'm="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[3:-1].split()
            [medications.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word)) in vocab]
        for term in re.finditer(r'do="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[4:-1].split()
            [dosages.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word) in vocab)]
        for term in re.finditer(r'mo="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[4:-1].split()
            [modes.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word) in vocab)]
        for term in re.finditer(r'f="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[3:-1].split()
            [frequencies.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word) in vocab)]
        for term in re.finditer(r'du="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[4:-1].split()
            [durations.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word) in vocab)]
        for term in re.finditer(r'r="[a-z0-9 ]+"', case.raw_labels):
            temp = term.group()[3:-1].split()
            [reasons.add(re.sub(r'\d+', '<num>', word)) for word in temp if (word not in stopwords) and (re.sub(r'\d+', '<num>', word) in vocab)]

    print('Number of: m={0}, do={1}, mo={2}, f={3}, du={4}, r={5}'.format(len(medications), len(dosages), len(modes), len(frequencies), len(durations), len(reasons)))
    return(medications, dosages, modes, frequencies, durations, reasons)

def visualise(model, sentences, labels, topn):
    words = [word for sent in sentences for word in sent]
    cnt = np.array(Counter(words).most_common(1000))
    topwords = np.ndarray.tolist(cnt[:, 0])

    visualisation = []
    colormap = []
    colors = ['red', 'green',  'yellow', 'purple', 'orange', 'cyan']

    [(visualisation.append(word), colormap.append(colors[i])) for i in range(len(labels)) for word in labels[i]]

    [(visualisation.append(word), colormap.append('blue')) for word in topwords if word not in visualisation]

    # This assumes words_top_ted is a list of strings, the top 1000 words
    words_vec = model[visualisation]

    tsne = TSNE(n_components=2, random_state=0)
    words_tsne = tsne.fit_transform(words_vec)

    p = figure(tools="pan,wheel_zoom,reset,save",
               toolbar_location="above",
               title="T-SNE for top " + str(topn) + " words and labels")

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