import os

import muspy
from tqdm import tqdm

dir = "/home/aik2/Learn/ComputerMusic/Models/REMI/remi_phrase_pop909/result/"
new_dir = "crop/"


def crop_first_bar(music: muspy.Music):
    one_bar = 16
    crop = muspy.Music(resolution=4)
    for inst in music.tracks:
        crop_track = muspy.Track()
        for note in inst.notes:
            if note.time >= one_bar:
                if note.duration == 0:
                    note.duration = 1
                crop_track.notes.append(
                    note.deepcopy().adjust_time(lambda x: x - one_bar)
                )
        crop.tracks.append(crop_track)
    return crop


if __name__ == "__main__":
    if os.path.exists(dir + new_dir):
        os.system(f"rm -rf {dir + new_dir}")
    os.makedirs(dir + new_dir)

    for midi in tqdm(os.listdir(dir)):
        if midi.endswith(".mid"):
            music = muspy.read_midi(dir + midi)
            music.adjust_resolution(4)
            music = crop_first_bar(music)
            music.write_midi(dir + new_dir + midi)
