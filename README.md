# Sonia
sensuous original network invents audio

FOR UBUNTU:

To create samples:
You need generation_script folder
  1. ```chmod u+x setup```
  2. ```./setup```
  3. ```python3 run.py```
  4. follow instructions
  5. sample will be created in samples folder
  
To train:
You need model_v3 folder
1. start Cr_Datav2.ipynb with jupyter
2. add path to your midi samples and save path
3. choose middle tempo of your samples, velocites that is usable in your .midi samples and times
4. run all 

5. start Midi_hitsv2.ipynb with jupyter
6. put times, velocities and tempo that you used in Cr_Datav2
7. put folder to pwd that you used as save folder in Cr_Datav2
8. run all
