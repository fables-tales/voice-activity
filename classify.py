import wave
import numpy
import struct
import sqlite3
import time

import scipy

import matplotlib.pyplot as pyplot
import cPickle as pickle

from sklearn.ensemble import RandomForestClassifier
from vadutils import *
from sklearn.cross_validation import cross_val_score, StratifiedKFold

PLOTTING = True

if PLOTTING:
    pyplot.ion()

def plot_spectrum(samples, freq, klass):
    if PLOTTING:
        n = len(samples)
        k = scipy.arange(n)
        T = n/freq
        frq = k/T
        frq = frq[range(n/2)]
        Y = numpy.fft.fft(samples)
        Y = Y[range(n/2)]
        if klass == "voice":
            color = "b"
        elif klass == "keyboard":
            color = "g"
        else:
            color = "r"
        pyplot.plot(frq, abs(Y), color, alpha=0.5)
        pyplot.xlabel("Freq (HZ)")
        pyplot.ylabel("Y(freq)")

def load_data():
    db = sqlite3.connect("db.sqlite")
    cur = db.cursor()
    voice_samples    = []
    keyboard_samples = []
    noise_samples = []
    if PLOTTING:
        pyplot.figure()
    for row in cur.execute("SELECT * FROM samples where voice=1"):
        load_vectors(row[0], voice_samples, row[1], row[2], "voice")

    for row in cur.execute("SELECT * FROM samples where keyboard=1"):
        load_vectors(row[0], keyboard_samples, row[1], row[2], "keyboard")

    for row in cur.execute("SELECT * FROM samples where keyboard=0 and voice=0"):
        load_vectors(row[0], noise_samples, row[1], row[2], "noise")
    if PLOTTING:
        pyplot.draw()
        pyplot.show()

    return voice_samples, keyboard_samples, noise_samples

def make_classifier():
    c = RandomForestClassifier(n_estimators=100, verbose=0, n_jobs=3)
    return c

def train_classifier(positive_vectors, negative_vectors, cross_val=False):
    train_attrs = positive_vectors + negative_vectors
    train_labels = [1]*len(positive_vectors) + [0]*len(negative_vectors)

    c = make_classifier()
    c.fit(train_attrs, train_labels)
    if cross_val:
        n = 10
        cv = StratifiedKFold(train_labels, k=n)
        vals = numpy.array(cross_val_score(make_classifier(), train_attrs, train_labels, cv=cv))

        print
        print  "cross val score",vals.mean(),vals.std()
        print

    return c

def accumulate_errors(expected, actual, errors=0):
    assert len(expected) == len(actual)

    for i in xrange(len(expected)):
        errors += 1 if expected[i] != actual[i] else 0

    return errors

def errors(classifier, vectors, klass):
    actual   = classifier.predict(vectors)
    expected = [klass for x in vectors]

    return accumulate_errors(expected, actual)

def make_predictions(classifier1, classifier2, samples):
    preds1 = classifier1.predict_proba(samples)
    preds2 = classifier2.predict_proba(samples)

    return [x[1] for x in preds1], [x[1] for x in preds2]

def show_separation(classifier1, classifier2, positive1, positive2, negative):
    if PLOTTING:
        preds_p1 = make_predictions(classifier1, classifier2, positive1)
        preds_p2 = make_predictions(classifier1, classifier2, positive2)
        preds_n  = make_predictions(classifier1, classifier2, negative)


        pyplot.figure()

        pyplot.scatter(preds_p1[1], preds_p1[0],c='b')
        pyplot.scatter(preds_p2[1], preds_p2[0], c='g')
        pyplot.scatter(preds_n[1], preds_n[0], c='r')
        pyplot.xlim([0,1])
        pyplot.ylim([0,1])
        pyplot.xlabel("P(voice)")
        pyplot.ylabel("P(keyboard)")
        pyplot.draw()
        pyplot.show()

if __name__ == "__main__":
    print "loading data"
    voice_vectors, keyboard_vectors, noise_vectors = load_data()
    print "training classifer"
    voice_negative    = noise_vectors + keyboard_vectors
    keyboard_negative = noise_vectors + voice_vectors
    voice_classifier   = train_classifier(voice_vectors, voice_negative)
    keyboard_classifier = train_classifier(keyboard_vectors, keyboard_negative)
    with open("classifier.pickle", "w") as packfile:
        result = [voice_classifier, keyboard_classifier]
        packfile.write(pickle.dumps(result))
        print "done pickling"

    print "testing classifier"

    #test labels against results
    s = time.time()
    print "voice positive errors", errors(voice_classifier, voice_vectors, 1)
    ke = errors(voice_classifier, keyboard_vectors, 0)
    print "voice negative (keyboard) errors", ke, ke*1.0/len(keyboard_vectors)
    print "voice negative (noise) errors", errors(voice_classifier, noise_vectors, 0)

    print "keyboard positive errors", errors(keyboard_classifier, keyboard_vectors, 1)
    print "keyboard negative (voice) errors", errors(keyboard_classifier, voice_vectors, 0)
    print "voice negative (noise) errors", errors(keyboard_classifier, noise_vectors, 0)
    e = time.time()
    print "voice vectors", len(voice_vectors)
    print "keyboard vectors", len(keyboard_vectors)
    print "noise vectors", len(noise_vectors)
    print

    show_separation(voice_classifier, keyboard_classifier, voice_vectors, keyboard_vectors, noise_vectors)
    if PLOTTING:
        raw_input("hit enter to ragequit! ")
