import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
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


def compute_cos_sim(music, phrase_config, is_melody, note_tracks=[0], chd_tracks=[1]):
    """
    CORE FUNCTION: Compute ILS (see WholeSongGen)

    Compute a pairwise similarity matrix of 2-measure segments within a
    song, ILS is defined as the ratio between same-type phrase similarity and global
    average similarity, and therefore higher values indicate better structure.
    """

    # compute the total number of bars
    num_bar = 0
    for phrase in phrase_config:
        assert phrase[1] % 2 == 0
        num_bar += phrase[1]
    num_2bar = num_bar // 2

    nmat = utils.get_note_matrix(music, note_tracks)

    # get phrases (type, start_bar, end_bar)
    phrases = []
    bar = 0
    for phrase in phrase_config:
        phrases.append((phrase[0], bar // 2, (bar + phrase[1]) // 2))
        bar += phrase[1]

    if is_melody:
        # use ec2vae
        prmat_ec2vae = utils.namt_to_prmat_ec2vae(nmat, num_2bar)
        if len(chd_tracks) > 0:
            chd_ec2vae = utils.get_chord_ec2vae(music, num_2bar, chd_tracks)
            dist_p, dist_r = ec2vae_enc.encoder(prmat_ec2vae, chd_ec2vae)
            zs = [dist_p.mean, dist_r.mean]
        else:
            raise RuntimeError

    else:
        # use polydis
        prmat = utils.nmat_to_prmat(nmat, num_2bar)  # (num_2bar, 32, 128)
        z_txt = txt_enc.forward(prmat).mean
        # utils.prmat_to_midi_file(prmat, "./prmat.mid")
        if len(chd_tracks) > 0:
            chd = utils.get_chord(music, num_2bar, chd_tracks)
            # utils.chd_to_midi_file(chd, "./chd.mid")
            z_chd = chd_enc.forward(chd).mean
            zs = [z_chd, z_txt]
        else:
            # print("NO CHORD! ONLY TXT")
            zs = [z_txt]

    ILS = []
    for z in zs:
        # Compute similarity matrix
        bs = z.shape[0]
        size = z.shape[1]
        z_sim_mat = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)
        # utils.show_matrix(z_sim_mat)
        for i in range(bs):
            assert z_sim_mat[i, i] == 1.0
        sim_sum = torch.sum(z_sim_mat)
        total_num = bs * bs

        # Compute ILS
        same_type = 0.0
        s_sum = 0
        for i, (tp_i, start_i, end_i) in enumerate(phrases):
            for j, (tp_j, start_j, end_j) in enumerate(phrases):
                if tp_i in ["A", "B"] and tp_i == tp_j and i != j:
                    same_type += torch.sum(z_sim_mat[start_i:end_i, start_j:end_j])
                    s_sum += (end_i - start_i) * (end_j - start_j)
        if s_sum == 0:
            return None
        same_type = (same_type / s_sum) / (sim_sum / total_num)
        ILS.append(float(same_type))

    return ILS


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


def single_midi(midi, is_melody, note_tracks, chd_tracks, phrase):
    assert phrase is not None
    phrase_config = utils.phrase_config_from_string(phrase)
    music = utils.get_music(dir)
    print(compute_cos_sim(music, phrase_config, is_melody, note_tracks, chd_tracks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", help="single generated MIDI file")
    parser.add_argument("--midi-dir", help="directory of generated MIDI files")
    parser.add_argument("--is-melody", action="store_true")
    parser.add_argument("--phrase", help="phrase configuration, eg. i4A8B8o4")
    parser.add_argument("--note-track", default=0)
    parser.add_argument("--chd-track", default=1)
    args = parser.parse_args()

    if args.midi is not None:
        single_midi(
            args.midi,
            args.is_melody,
            [int(args.note_track)],
            [int(args.chd_track)],
            args.phrase,
        )
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
