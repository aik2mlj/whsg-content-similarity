import argparse
import os

import muspy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from chord_extractor import extract_chords_from_midi_file
from pretrained_encoders import (
    Ec2VaeEncoder,
    PolydisChordEncoder,
    PolydisTextureEncoder,
)

ec2vae_enc = Ec2VaeEncoder.create_2bar_encoder()
chd_enc = PolydisChordEncoder.create_encoder()
txt_enc = PolydisTextureEncoder.create_encoder()
ec2vae_enc.load_state_dict(torch.load("./pretrained_models/ec2vae_enc_2bar.pt"))
chd_enc.load_state_dict(torch.load("./pretrained_models/polydis_chd_enc.pt"))
txt_enc.load_state_dict(torch.load("./pretrained_models/polydis_txt_enc.pt"))


def compute_cos_sim(music: muspy.Music, chords, is_melody):
    """
    CORE FUNCTION: Compute ILS (see WholeSongGen)

    Compute a pairwise similarity matrix of 2-measure segments within a
    song, ILS is defined as the ratio between same-type phrase similarity and global
    average similarity, and therefore higher values indicate better structure.
    """

    # compute the total number of bars
    num_bar = len(music.barlines)
    num_2bar = num_bar // 2
    print(num_bar)

    nmat = utils.get_note_matrix(music)
    nmat = utils.dedup_note_matrix(nmat)

    if is_melody:
        # use ec2vae
        prmat_ec2vae = utils.namt_to_prmat_ec2vae(nmat, num_2bar)
        if chords is not None:
            chd_ec2vae = utils.get_chord_ec2vae(num_2bar, chords)
            dist_p, dist_r = ec2vae_enc.encoder(prmat_ec2vae, chd_ec2vae)
            zs = [dist_p.mean, dist_r.mean]
        else:
            raise RuntimeError

    else:
        # use polydis
        prmat = utils.nmat_to_prmat(nmat, num_2bar)  # (num_2bar, 32, 128)
        z_txt = txt_enc.forward(prmat).mean
        # utils.prmat_to_midi_file(prmat, "./prmat.mid")
        if chords is not None:
            chd = utils.get_chord(num_2bar, chords)
            # utils.chd_to_midi_file(chd, "./chd.mid")
            z_chd = chd_enc.forward(chd).mean
            zs = [z_chd, z_txt]
        else:
            # print("NO CHORD! ONLY TXT")
            zs = [z_txt]

    avg = []
    for z in zs:
        # Compute similarity matrix
        bs = z.shape[0]
        size = z.shape[1]
        z_sim_mat = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)
        utils.show_matrix(z_sim_mat)
        for i in range(bs):
            assert z_sim_mat[i, i] == 1.0
        sim_sum = torch.sum(z_sim_mat)
        total_num = bs * bs

        avg.append(float(sim_sum / total_num))

    return avg


def ground_truth(is_melody, note_tracks, chd_tracks):
    midi_dpath = "data/matched_pop909_acc"
    midi_fname = "aligned_demo.mid"
    anno_dpath = "data/original_shuqi_data"
    anno_fname = "human_label1.txt"

    # songs that have even number of phrases
    good_songs = []
    for i in range(1, 909 + 1):
        anno_fpath = f"{anno_dpath}/{i:03}/{anno_fname}"
        with open(anno_fpath) as f:
            phrase_annotation = f.readline().strip()
        phrase_config = utils.phrase_config_from_string(phrase_annotation)
        bad = False
        for phrase in phrase_config:
            if phrase[1] % 2 != 0:
                bad = True
                break
        if not bad:
            good_songs.append(i)
    print(len(good_songs))

    songs = good_songs
    s = []
    for song in tqdm(songs):
        midi_fpath = f"{midi_dpath}/{song:03}/{midi_fname}"
        anno_fpath = f"{anno_dpath}/{song:03}/{anno_fname}"
        with open(anno_fpath) as f:
            phrase_annotation = f.readline().strip()
        phrase_config = utils.phrase_config_from_string(phrase_annotation)
        music = utils.get_music(midi_fpath)
        same_types = compute_cos_sim(
            music, phrase_config, is_melody, note_tracks, chd_tracks
        )
        if same_types is not None:
            s.append(same_types)
    print(len(s))
    s = np.array(s)
    s_mean = s.mean(axis=0)
    s_std = s.std(axis=0)
    print(s_mean, s_std)


def from_dir(dir, is_melody, note_tracks, chd_tracks):
    s = []
    for phrase_anno in os.scandir(dir):
        if phrase_anno.is_dir():
            phrase_config = utils.phrase_config_from_string(phrase_anno.name)
            print(phrase_config)
            for f in tqdm(os.scandir(phrase_anno.path)):
                fpath = f.path
                music = utils.get_music(fpath)
                same_types = compute_cos_sim(
                    music, phrase_config, is_melody, note_tracks, chd_tracks
                )
                s.append(same_types)
    s = np.array(s)
    s_mean = s.mean(axis=0)
    s_std = s.std(axis=0)
    print(s_mean, s_std)


def single_midi(midi, is_melody, chords=None):
    music = utils.get_music(midi)
    print(compute_cos_sim(music, None, is_melody))


if __name__ == "__main__":
    """
    MIDIs should be 2 tracks: Content (--note-track) & Chords (--chd-track)
        Chords: 36-48 is bass, 48-60 is chroma
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", help="single generated MIDI file")
    parser.add_argument("--midi-dir", help="directory of generated MIDI files")
    parser.add_argument(
        "--is-melody",
        action="store_true",
        help="If this is monophonic melody (use EC2VAE), otherwise polyphonic accompaniment (use Polydis)",
    )
    parser.add_argument("--phrase", help="phrase configuration, eg. i4A8B8o4")
    parser.add_argument("--note-track", default=0)
    parser.add_argument("--chd-track", default=1)
    args = parser.parse_args()

    if args.midi is not None:
        single_midi(args.midi, args.is_melody)
        exit(0)

    # midi_dir = "./generated/128samples/ours/mel+acc_samples"
    assert args.midi_dir is not None
    from_dir(
        args.midi_dir,
        args.is_melody,
        [int(args.note_track)],
        [int(args.chd_track)],
    )
    ground_truth(args.is_melody, [0], [3])
