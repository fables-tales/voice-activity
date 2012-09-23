import wave
import struct
import os
from sklearn.ensemble import RandomForestClassifier
import time

def wave_file_to_samples(filename):
    waveobj = wave.open(filename)
    frames = []
    for i in xrange(0,waveobj.getnframes()):
        frames.append(struct.unpack("h", waveobj.readframes(1))[0])

    return frames

def samples_to_16ms_frames(samples, framerate=44100):
    frame_size = int(16/1000.0 * framerate)
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

def load_vectors(filename, vector_group):
    samples = wave_file_to_samples(filename)
    frames = samples_to_16ms_frames(samples)
    vector_group += frames


def load_data():
    noise_files = ["noise_samples/" + x for x in os.listdir("noise_samples")]
    voice_files = ["voice_samples/" + x for x in os.listdir("voice_samples")]


    noise_vectors = []
    for item in noise_files:
        load_vectors(item, noise_vectors)

    voice_vectors = []
    for item in voice_files:
        load_vectors(item, voice_vectors)

    return voice_vectors, noise_vectors

def train_classifier(voice_vectors, noise_vectors):
    train_attrs = voice_vectors + noise_vectors
    train_labels = [1]*len(voice_vectors) + [0]*len(noise_vectors)

    c = RandomForestClassifier(n_estimators=50, verbose=0, n_jobs=1)
    c.fit(train_attrs, train_labels)

    return c

if __name__ == "__main__":
    print "loading data"
    voice_vectors, noise_vectors = load_data()
    print "training classifer"
    c = train_classifier(voice_vectors, noise_vectors)
    print "testing classifier"


    #test labels against results
    wrong = 0
    s = time.time()
    for item in voice_vectors:
        pred = c.predict(item)
        if pred != 1:
            wrong += 1

    for item in noise_vectors:
        pred = c.predict(item)
        if pred != 0:
            wrong += 1

    e = time.time()
    classifies = len(voice_vectors) + len(noise_vectors)
    cps = classifies*1.0/(e-s)
    print "results :)"
    print
    print "%18s %14s %15s %25s %12s" % ("classified wrong", "classified", "error rate", "classifies per second", "overspeed")
    print "%18d %14d %15.3f %25.3f %12.3f" % (wrong, classifies, wrong*100.0/classifies, cps, cps/60)
