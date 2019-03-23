import os
from generator import generate
from constants import pwd
from midi2audio import FluidSynth
from constants import save_pwd
# finding models
models = sorted([f for f in os.listdir(pwd) if f[len(f)-8:len(f)] != "_classes"])
def main():
    print("Choose Model\n type it's number")
    for num, model in enumerate(models):
        print(num, model)
    model = int(input())
    
    d = False
    if (model == -1):
        d = True
        print("Secret_code_dudec_activated")
        print("Choose Model\n type it's number")
        model = int(input())
#   testing is number correct
    if (model >= len(models) or model < 0):
        print()
        return 1
    
    print()
    print("type integer seed")
    seed = int(input())
    
#     generating midi
    midi = generate(models[model], seed=seed, d=d)
    
#    creating wav
    fs = FluidSynth(os.getcwd() + '/GeneralUser_GS_SoftSynth_v144.sf2')
    fs.midi_to_audio(save_pwd+'sample.mid', save_pwd +'output.wav')
    
    print()
    print("Length of the song approximately = " + str(midi.length))
    return 0

while(main()):
    print("Try one more time")
