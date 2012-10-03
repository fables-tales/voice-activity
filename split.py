import wave
import struct
from math import ceil, floor
import sqlite3

def extract_sample(waveobj, start_frame, end_frame):
    assert start_frame < end_frame
    waveobj.setpos(start_frame)

    framevalues = []

    for i in xrange(start_frame, end_frame):
        frame = waveobj.readframes(1)
        data_val = struct.unpack( "h", frame)[0]
        framevalues.append(data_val)

    return framevalues

def extract_sample_seconds(waveobj, start_seconds, end_seconds):
    assert start_seconds < end_seconds
    rate = waveobj.getframerate()
    start_samples = int(floor(start_seconds*rate))
    end_samples   = int(ceil(end_seconds*rate))
    return extract_sample(waveobj, start_samples, end_samples)

def write_samples_to_file(filename, samples):
    waveobj = wave.open(filename, "w")
    waveobj.setnchannels(1)
    waveobj.setsampwidth(2)
    waveobj.setframerate(44100)

    for sample in samples:
        waveobj.writeframes(struct.pack("h", sample))

    waveobj.close()

def make_audio_slice(waveobj, filename, start_seconds, end_seconds):
    samples = extract_sample_seconds(waveobj, start_seconds, end_seconds)

if __name__ == "__main__":
    db = sqlite3.connect("db.sqlite")
    cur = db.cursor()
    for idx,row in enumerate(cur.execute("select * from samples")):
        print idx
        waveobj             = wave.open("source_audio/" + row[0])
        start_seconds       = row[1]
        end_seconds         = row[2]
        if end_seconds == -1:
            end_seconds = waveobj.getnframes()*1.0/waveobj.getframerate()
        print end_seconds
        voice               = bool(row[3])
        keyboard            = bool(row[4])
        slice = make_audio_slice(waveobj, "/tmp/result." + str(idx) + ".wav", start_seconds, end_seconds)
