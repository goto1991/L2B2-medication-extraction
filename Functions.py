import os
import lxml.etree
import zipfile
import re
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
import random
import tensorflow as tf

from Set import pool
from DS import DS
from Iterator import Iterator

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
#from bokeh.io import output_notebook
#£output_notebook()


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

    vocab = {word for sent in Dataset.get_sentences() for word in sent}
    labelled = Dataset.get_DS(labelled='yes')

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


def visualise(model, sentences, labels, topn, title):
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


def get_traintest(model, labels, train_size, test_size, train_label_percentage, test_label_percentage, word_repetition, label_repetition):
    train_set = []
    train_labels = []
    train_size = train_size
    test_set = []
    test_labels = []
    test_size = test_size

    vocab = set(model.wv.vocab.keys())
    target = set(labels)

    for i in range(train_size * (100 - train_label_percentage) // 100):
        word = random.sample(vocab, 1)
        train_set.append(model[word[0]])
        if not word_repetition: vocab.discard(word[0])
        if word[0] in target:
            train_labels.append([1, 0])
            if not label_repetition: target.discard(word[0])
        else:
            train_labels.append([0, 1])
    for i in range(train_size * train_label_percentage // 100):
        word = random.sample(target, 1)
        train_set.append(model[word[0]])
        train_labels.append([1, 0])
        if not label_repetition: target.discard(word[0])

    for i in range(test_size * (100 - test_label_percentage) // 100):
        word = random.sample(vocab, 1)
        test_set.append(model[word[0]])
        if not word_repetition: vocab.discard(word[0])
        if word[0] in labels:
            test_labels.append([1, 0])
            if not label_repetition: target.discard(word[0])
        else:
            test_labels.append([0, 1])

    for i in range(test_size * test_label_percentage // 100):
        word = random.sample(target, 1)
        test_set.append(model[word[0]])
        test_labels.append([1, 0])
        if not label_repetition: target.discard(word[0])

    print('Percentage of labels in train set: %f' %(np.array(train_labels).sum(0)[0]/len(train_labels)))
    print('Percentage of labels in test: %f' %(np.array(test_labels).sum(0)[0] / len(test_labels)))
    return train_set, train_labels, test_set, test_labels


def model_1(train_set, train_labels, test_set, test_labels, target='medications', repetitions='no ', report_percentage=20, epochs=1000, batch=50, input_size=100, output_size=2, node_count=50):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.05)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    node_count = node_count

    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])

    # Define the first layer here
    W = weight_variable([input_size, node_count])
    b = bias_variable([node_count])
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

    # Use dropout for this layer (should you wish)
    # keep_prob = tf.placeholder(tf.float32)
    # h_drop = tf.nn.dropout(h1, keep_prob)

    # Define the second layer here
    # W2 = weight_variable([node_count_1, node_count_2])
    # b2 = bias_variable([node_count_2])
    # h2 = tf.nn.sigmoid(tf.matmul(h, W) + b)

    # Define the output layer here
    V = weight_variable([node_count, output_size])
    c = bias_variable([output_size])
    y = tf.nn.softmax(tf.matmul(h, V) + c)

    # We'll use the cross entropy loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # And classification accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # And the Adam optimiser
    train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cross_entropy)

    # Start a tf session and run the optimisation algorithm
    sess = tf.Session()
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())

    training = Iterator(train_set, train_labels)
    N = 0

    while training.epochs < epochs:
        trd, trl = training.next_batch(batch)
        report_mark = epochs * batch * report_percentage // 100
        if N % report_mark == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: trd, y_: trl})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_set, y_: test_labels})
            print("%d%% %s with %srepetitions training complete, training accuracy: %f, test accuracy: %f" % (N * report_percentage // report_mark, target, repetitions, train_accuracy, test_accuracy))
        sess.run(train_step, feed_dict={x: trd, y_: trl})
        N += 1

    print("Final Test Accuracy: %f\n" % (sess.run(accuracy, feed_dict={x: test_set, y_: test_labels})))