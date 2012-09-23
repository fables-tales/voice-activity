import wave
import struct
from math import ceil, floor
import sys

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
    write_samples_to_file(filename, samples)



if __name__ == "__main__":
    waveobj             = wave.open(sys.argv[1])
    voice_start_seconds = float(sys.argv[2])
    voice_end_seconds   = float(sys.argv[3])
    audio_end_seconds   = waveobj.getnframes()*1.0/waveobj.getframerate()

    make_audio_slice(waveobj, "noise_samples/1.wav", 0, voice_start_seconds)
    make_audio_slice(waveobj, "noise_samples/2.wav", voice_end_seconds, audio_end_seconds)

    make_audio_slice(waveobj, "voice_samples/1.wav", voice_start_seconds, voice_end_seconds)
