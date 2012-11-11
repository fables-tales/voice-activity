from __future__ import division
import time
import sys
import wave
import vadutils
import subprocess
import numpy
import sqlite3
import cPickle as pickle

SILENCE_THRESHOLD = (1<<24)*0.02
SILENCE = 1
SOUND   = 2

WINDOW_WIDTH = int(48000*0.2)

def is_frame_silent(frame):
    return abs(frame) < SILENCE_THRESHOLD

def current_time(wave_reader):
    samplerate = wave_reader.getframerate()
    return wave_reader.tell()/samplerate

def wind_region(wave_reader):
    values = vadutils.getframes(wave_reader, WINDOW_WIDTH)

    mode = 0
    if is_frame_silent(numpy.sum(values)):
        mode = SILENCE
    else:
        mode = SOUND

    original_mode = mode

    while mode == original_mode:
        values = vadutils.getframes(wave_reader, WINDOW_WIDTH)

        if is_frame_silent(numpy.sum(values)):
            mode = SILENCE
        else:
            mode = SOUND
    pos = wave_reader.tell()
    wave_reader.setpos(pos-WINDOW_WIDTH)

def find_endpoints(wave_reader):
    start_time = current_time(wave_reader)
    wind_region(wave_reader)
    end_time = current_time(wave_reader)
    return start_time, end_time

def invoke_mplayer(filename, start_time, length):
    target_filename = "/tmp/testing.wav"
    vadutils.cut_region(filename, start_time, length, target_filename)

    command = ["mplayer", target_filename]

    p = subprocess.Popen(command)
    p.wait()

def play_region(filename, start_time, end_time):
    length = end_time - start_time
    invoke_mplayer(filename, start_time, length)

def insert_sample(filename, start_time, end_time, voice, keyboard):
    query = "INSERT INTO samples VALUES (?,?,?,?,?)"
    conn = sqlite3.connect("db.sqlite")
    cur = conn.cursor()
    cur.execute(query, (filename, start_time, end_time, voice, keyboard))
    conn.commit()

def current_end_of_file(filename):
    query = "SELECT max(endtime) from samples where sourcefile=?"
    conn = sqlite3.connect("db.sqlite")
    cur = conn.cursor()
    cur.execute(query, (filename,))
    v = cur.next()[0]
    if v == None:
        return 0
    else:
        return v

def sample_end_time(wave_reader):
    return wave_reader.getnframes()*1.0/wave_reader.getframerate()

def main():
    filename = sys.argv[1]
    wave_reader = wave.open(filename)
    voice_classifier,keyboard_classifier = pickle.load(open("classifier.pickle"))
    try:
        wave_reader.setpos(int(current_end_of_file(filename)*48000)+1)
    except:
        print "file done"
        return
    print "bees"
    print current_time(wave_reader)
    start_time, end_time = find_endpoints(wave_reader)
    print "bees2"
    while start_time < current_end_of_file(filename):
        assert False
        start_time, end_time = find_endpoints(wave_reader)
    print start_time
    for i in range(0,50):
        print start_time, end_time
        if numpy.sum(vadutils.read_region(filename, start_time, end_time-start_time)) == 0:
            print "silence detected!"
            voice = False
            keyboard = False
        else:
            play_region(filename, start_time, end_time)
            vectors = []
            vadutils.load_vectors(filename, vectors,start_time, end_time)
            print numpy.average(voice_classifier.predict_proba(vectors), axis=0)
            voice = raw_input("was that voice? [y/n] ") == "y"
            keyboard = raw_input("was that keyboard? [y/n] ") == "y"

        insert_sample(filename, start_time, end_time, voice, keyboard)
        start_time, end_time = find_endpoints(wave_reader)
        if abs(current_time(wave_reader)-sample_end_time(wave_reader)) <= 0.5:
            break

if __name__ == "__main__":
    main()
