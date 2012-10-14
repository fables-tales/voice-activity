import wave
import numpy
import struct
import sqlite3
import time

import scipy

import matplotlib.pyplot as pyplot

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold


def get_samples(filename, start_seconds, end_seconds):
    waveobj = wave.open("source_audio/" + filename)

    start_samples = int(start_seconds * waveobj.getframerate())
    if end_seconds != -1:
        end_samples = end_seconds * waveobj.getframerate()
    else:
        end_samples = waveobj.getnframes()

    end_samples = int(end_samples)

    frames = []
    waveobj.setpos(start_samples)
    for i in xrange(start_samples, end_samples):
        frames.append(struct.unpack("h", waveobj.readframes(1))[0])

    return frames

window_width_ms = 16

def samples_to_16ms_frames(samples, framerate=44100):
    frame_size = int(window_width_ms/1000.0 * framerate)
    frames = []
    build = []
    ptr = 0

    while ptr < len(samples):
        build.append(samples[ptr])
        ptr += 1
        if ptr % frame_size == 0:
            frames.append(build)
            build = []

    if len(build) < frame_size:
        length_diff = frame_size - len(build)
        build += [0]*length_diff

    frames.append(build)

    return frames

def n_zero_crossings(values):
    n = 0
    for i in xrange(1,len(values)):
        sign1 = -1 if values[i-1] < 0 else 1
        sign2 = -1 if values[i] < 0 else 1
        if sign1 != sign2:
            n += 1

    return n

def load_vectors(filename, vector_group, start, end):
    samples = get_samples(filename, start, end)
    frames = samples_to_16ms_frames(samples)
    ffts = numpy.abs(numpy.fft.fft(frames))
    features = []
    for i in range(0,len(frames)):
        features.append(list(ffts[i]) + [n_zero_crossings(frames[i])])
    vector_group += features

def load_data():
    db = sqlite3.connect("db.sqlite")
    cur = db.cursor()
    voice_samples    = []
    keyboard_samples = []
    noise_samples = []
    for row in cur.execute("SELECT * FROM samples where voice=1"):
        load_vectors(row[0], voice_samples, row[1], row[2])

    for row in cur.execute("SELECT * FROM samples where keyboard=1"):
        load_vectors(row[0], keyboard_samples, row[1], row[2])

    for row in cur.execute("SELECT * FROM samples where keyboard=0 and voice=0"):
        load_vectors(row[0], noise_samples, row[1], row[2])

    return voice_samples, keyboard_samples, noise_samples


def make_classifier():
    c = RandomForestClassifier(n_estimators=100, verbose=0, n_jobs=1)
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

    print preds1.shape
    print len(samples)
    return [x[1] for x in preds1], [x[1] for x in preds2]

def show_separation(classifier1, classifier2, positive1, positive2, negative):
    preds_p1 = make_predictions(classifier1, classifier2, positive1)
    preds_p2 = make_predictions(classifier1, classifier2, positive2)
    preds_n  = make_predictions(classifier1, classifier2, negative)


    plt.scatter(preds_p1[1], preds_p1[0])
    plt.scatter(preds_p2[1], preds_p2[0], c='g')
    plt.scatter(preds_n[1], preds_n[0], c='r')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

if __name__ == "__main__":
    print "loading data"
    voice_vectors, keyboard_vectors, noise_vectors = load_data()
    print "training classifer"
    voice_negative    = noise_vectors + keyboard_vectors
    keyboard_negative = noise_vectors + voice_vectors
    voice_classifier   = train_classifier(voice_vectors, voice_negative)
    keyboard_classifier = train_classifier(keyboard_vectors, keyboard_negative)

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
    print len(voice_vectors) + len(noise_vectors) + len(keyboard_vectors)
    show_separation(voice_classifier, keyboard_classifier, voice_vectors, keyboard_vectors, noise_vectors)

