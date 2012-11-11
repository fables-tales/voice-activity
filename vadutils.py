from __future__ import division
import struct
import wave
import numpy

def decode_frame(wave_reader, frame):
    width = wave_reader.getsampwidth()
    if width == 2:
        frame = "\x00" + frame
    negative = struct.unpack("b", frame[-1])[0] < 0
    padding = "\xff" if negative else "\x00"
    value = struct.unpack("<i", frame + padding)[0]
    return value

def getframe(wave_reader):
    frame = wave_reader.readframes(1)
    return decode_frame(wave_reader, frame)

def getframes(wave_reader, nframes):
    buf = wave_reader.readframes(nframes)
    width = wave_reader.getsampwidth()
    frames = [decode_frame(wave_reader, buf[i:i+width]) for i in xrange(0,len(buf),width)]

    return frames

def cut_region(in_file_name, start_time, length, out_file_name):
    wave_reader = wave.open(in_file_name)
    tf = open(out_file_name, "w")
    wave_writer = wave.open(tf, "w")
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(3)
    wave_writer.setframerate(wave_reader.getframerate())

    wave_reader.setpos(int(start_time*wave_reader.getframerate()))

    frames = getframes(wave_reader, int(length*wave_reader.getframerate()))
    outframes = [struct.pack("i", frame)[0:3] for frame in frames]
    wave_writer.writeframes("".join(outframes))

    wave_writer.close()

def compute_end_samples_seconds(waveobj, end_seconds):
    if end_seconds != -1:
        end_samples = end_seconds * waveobj.getframerate()
    else:
        end_samples = waveobj.getnframes()
        end_seconds = end_samples / waveobj.getframerate()

    return end_samples,end_seconds

def get_samples(filename, start_seconds, end_seconds):
    #open a wave reader, get the sample width
    waveobj = wave.open(filename)

    start_samples = int(start_seconds * waveobj.getframerate())
    end_samples,end_seconds = compute_end_samples_seconds(waveobj, end_seconds)
    end_samples = int(end_samples)

    frames = []
    waveobj.setpos(start_samples)
    for i in xrange(start_samples, end_samples):
        value = getframe(waveobj)
        frames.append(value)

    return frames

window_width_ms = 16

def samples_to_16ms_frames(samples, framerate=48000):
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


def get_16_ms_frames(filename, start_seconds, end_seconds):
    return samples_to_16ms_frames(get_samples(filename, start_seconds, end_seconds))

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
