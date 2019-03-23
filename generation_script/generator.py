from io import open
import torch
import numpy as np
from mido import Message, MetaMessage
import mido
import random
import torch.nn as nn
import torch.nn.functional as F
from constants import pwd, save_pwd
max_len = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, other):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru1 = nn.GRU(classes, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.gru3 = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.velocity = other[0]
        self.time_range = other[1]
        self.tempo = other[2]
        self.pedal_max = other[3]
        self.notes = other[4]

    def forward(self, input, hidden):
        output, hidden[0] = self.gru1(input, hidden[0])
        output = F.relu(output)
        output, hidden[1] = self.gru2(output, hidden[1])
        output = F.relu(output)
        output, hidden[2] = self.gru3(output, hidden[2])
        output = self.softmax(self.out(output[0]))

        return output, hidden


def list_find(l, n):
    for i in range(len(l)):
        if l[i] == n:
            return i
    return -1



def class_to_msg(n):
    note = min(class_notes[n % len(class_notes)] + 5, 127)
    n //= len(class_notes)
    t = time_range[n % 3]
    n //= 3
    v = velocity_range[n]
    if (note < 60):
        v = v * 2 // 3
    return Message("note_on", note=note, velocity=v, time=t)




def list_to_midi(sample):
    midi = mido.MidiFile()
    tr = mido.MidiTrack()
    midi.tracks.append(tr)
    tr.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    if (control != None):
        tr.append(Message("program_change", channel=0, program=control,time=0))
    pedal = pedal_max
    last_notes = []

    for i in sample:
        if (use_pedal):
            if (pedal == 0):
                for j in range(len(last_notes) - 10):
                    tr.append(Message("note_off", note=last_notes[j], velocity=0, time=0))
                pedal = pedal_max
                last_notes = last_notes[len(last_notes) - 10 : len(last_notes)]
        tr.append(class_to_msg(i))
        if (use_pedal):
            pedal -= 1
            last_notes.append(class_notes[i % len(class_notes)])
    if (use_pedal):
        for i in last_notes:
            tr.append(Message("note_off", note=i, velocity=0, time=0))
    tr.append(MetaMessage("end_of_track", time=0))
    return midi


def one_hot(c):
    x = torch.zeros(classes, device=device, dtype=torch.float)
    x[c] = 1;
    return x.view(1, 1, -1)


def evaluate(decoder, num=-1, max_length=max_len, seed=None):
    with torch.no_grad():
        if (seed != None):
            random.seed(seed)
            np.random.seed(seed)
        decoder_input = one_hot(sos)
        decoder_hidden = []
        for j in range(3):
            decoder_hidden.append([random.random() * 10 for i in range(hidden_size)])
        decoder_hidden = [torch.tensor(decoder_hidden[0], device=device).view(1, 1, -1),
                          torch.tensor(decoder_hidden[1], device=device).view(1, 1, -1),
                          torch.tensor(decoder_hidden[2], device=device).view(1, 1, -1)]

        decoded_words = []

        for di in range(max_length):
            # if (seed != None):
            #     np.random.seed(seed + di)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(random.randint(3, 4))
            topv = np.exp(topv.cpu()[0].numpy())
            topv[1] += 0.2
            # topv = (-1)/topv.cpu()[0].numpy()
            summ = topv.sum()
            # print(topv / summ)
            chosen = np.random.choice(topi[0].cpu(), p=topv / summ)
            decoder_input = one_hot(chosen)
            decoded_words.append(chosen)

        return decoded_words


def create_dudec(length):
    tr = mido.MidiTrack()
    d = mido.MidiFile('../Dudets.mid')
    messages = []
    for track in d.tracks:
        for msg in track:
            if (msg.type == 'note_off' or msg.type == 'note_on'):
                msg.time = int(msg.time * 1.5)
                msg.velocity = 50
                messages.append(msg)
    for i in range(int(length) // int(d.length)):
        for i in messages:
            tr.append(i)
    return tr


def generate_sample(number, decoder, seed=None):
    for i in range(number):
        new_music = evaluate(decoder, max_length=1000, seed=seed)
        midi = list_to_midi(new_music)
        if (dudets):
            midi.tracks.append(create_dudec(midi.length))
        midi.save(save_pwd + "sample" + ".mid")
    return midi


def generate(model, seed=None,d=False):

    global dudets
    dudets = d
    global control
    control = None
    with open(pwd + model + '_classes') as f:
        file = f.readlines()[0].split()
        global classes, tempo, time_range, velocity_range, pedal_max, class_notes, sos, hidden_size, use_pedal
        classes = int(file[0])
        hidden_size = int(file[1])
    sos = classes - 1
    use_pedal = True
    map_location = 'cpu'
    decoder = DecoderRNN(hidden_size, classes, [0, 0, 0, 0, 0]).to(device)
    checkpoint = torch.load(pwd + model,map_location=map_location)
    class_notes = checkpoint['class_note']
    pedal_max = 20
    velocity_range = checkpoint['velocity_range']
    time_range = checkpoint['time_range']
    tempo = checkpoint['tempo']
    decoder.load_state_dict(checkpoint['model_state_dict'])
    return generate_sample(1, decoder, seed=seed)
