from __future__ import division
import time
import sys
import wave
from vadutils import getframe
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
    values = [0]*WINDOW_WIDTH
    for i in range(0,WINDOW_WIDTH):
        values[i] = abs(getframe(wave_reader))

    mode = 0
    if is_frame_silent(sum(values)/len(values)):
        mode = SILENCE
    else:
        mode = SOUND

    original_mode = mode

    while mode == original_mode:
        for i in range(0,WINDOW_WIDTH):
            values[i] = abs(getframe(wave_reader))

        if is_frame_silent(sum(values)/len(values)):
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

if __name__ == "__main__":
    filename = sys.argv[1]
    wave_reader = wave.open(filename)
    voice_classifier,keyboard_classifier = pickle.load(open("classifier.pickle"))
    start_time, end_time = find_endpoints(wave_reader)
    print current_end_of_file(filename)
    while start_time < current_end_of_file(filename):
        start_time, end_time = find_endpoints(wave_reader)
    print start_time
    for i in range(0,10):
        print start_time, end_time
        play_region(filename, start_time, end_time)
        vectors = []
        vadutils.load_vectors(filename, vectors,start_time, end_time)
        print numpy.average(voice_classifier.predict_proba(vectors), axis=0)
        voice = raw_input("was that voice? [y/n] ") == "y"
        keyboard = raw_input("was that keyboard? [y/n] ") == "y"
        insert_sample(filename, start_time, end_time, voice, keyboard)
        start_time, end_time = find_endpoints(wave_reader)
