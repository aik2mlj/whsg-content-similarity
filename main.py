import argparse

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

# Calculate cosine similarity
# cosine_sim = F.cosine_similarity(vector1, vector2)
# print("Cosine Similarity:", cosine_sim.item())

"""
MIDIs should be 2 tracks: Content & Chords
    Chords: 36-48 is bass, 48-60 is chroma
"""
ec2vae_enc = Ec2VaeEncoder.create_2bar_encoder()
chd_enc = PolydisChordEncoder.create_encoder()
txt_enc = PolydisTextureEncoder.create_encoder()
ec2vae_enc.load_state_dict(torch.load("./pretrained_models/ec2vae_enc_2bar.pt"))
chd_enc.load_state_dict(torch.load("./pretrained_models/polydis_chd_enc.pt"))
txt_enc.load_state_dict(torch.load("./pretrained_models/polydis_txt_enc.pt"))


def compute_cos_sim(music, phrase_config, is_melody, note_tracks=[0], chd_tracks=[1]):
    num_bar = 0
    for phrase in phrase_config:
        assert phrase[1] % 2 == 0
        num_bar += phrase[1]
    num_2bar = num_bar // 2

    nmat = utils.get_note_matrix(music, note_tracks)

    phrases = []  # (type, start_bar, end_bar)
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

    withins, same_types = [], []
    for z in zs:
        bs = z.shape[0]
        size = z.shape[1]
        z_sim_mat = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)
        # utils.show_matrix(z_sim_mat)
        for i in range(bs):
            assert z_sim_mat[i, i] == 1.0
        sim_sum = torch.sum(z_sim_mat)
        total_num = bs * bs

        # # within each phrase
        # within = 0.
        # within_num = 0
        # for tp, start, end in phrases:
        #     within += torch.sum(z_sim_mat[start: end, start: end])
        #     within_num += ((end - start)**2)
        # within = (within / within_num) / (sim_sum / total_num)
        # withins.append(float(within))

        # same phrase type
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
        same_types.append(float(same_type))

    # print(withins, same_types)
    return same_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi")
    parser.add_argument("--melody", action="store_true")
    parser.add_argument("--phrase", help="phrase configuration, eg. i4A8B8o4")
    parser.add_argument("--note-track", default=0)
    parser.add_argument("--chd-track", default=1)
    args = parser.parse_args()

    # dir = "./128samples/diff+phrase/mel+acc_samples"
    is_melody = True
    note_tracks = [0]
    chd_tracks = [3]

    # if os.path.isdir(dir):
    #     w, s = [], []
    #     for phrase_anno in os.scandir(dir):
    #         if phrase_anno.is_dir():
    #             phrase_config = utils.phrase_config_from_string(phrase_anno.name)
    #             print(phrase_config)
    #             for f in tqdm(os.scandir(phrase_anno.path)):
    #                 fpath = f.path
    #                 music = utils.get_music(fpath)
    #                 same_types = compute_cos_sim(music, phrase_config, is_melody, note_tracks, chd_tracks)
    #                 s.append(same_types)
    #     # w = np.array(w)
    #     # print(w.shape)  # (128, 2)
    #     # w_mean = w.mean(axis=0)
    #     # w_std = w.std(axis=0)
    #     s = np.array(s)
    #     s_mean = s.mean(axis=0)
    #     s_std = s.std(axis=0)
    #     # print(w_mean, w_std)
    #     print(s_mean, s_std)
    # else:
    #     phrase_config = utils.phrase_config_from_string(args.phrase)
    #     music = utils.get_music(dir)
    #     print(compute_cos_sim(music, phrase_config, is_melody, note_tracks, chd_tracks))

    midi_dpath = "/home/aik2/Learn/ComputerMusic/WholeSongGen/dataset-code-for-lejun/data/matched_pop909_acc"
    midi_fname = "aligned_demo.mid"
    anno_dpath = "/home/aik2/Learn/ComputerMusic/WholeSongGen/dataset-code-for-lejun/data/original_shuqi_data"
    anno_fname = "human_label1.txt"
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
        # print(phrase_config)
        music = utils.get_music(midi_fpath)
        same_types = compute_cos_sim(
            music, phrase_config, is_melody, note_tracks, chd_tracks
        )
        if same_types is not None:
            s.append(same_types)
        # w = np.array(w)
        # print(w.shape)  # (128, 2)
        # w_mean = w.mean(axis=0)
        # w_std = w.std(axis=0)
    print(len(s))
    s = np.array(s)
    s_mean = s.mean(axis=0)
    s_std = s.std(axis=0)
    # print(w_mean, w_std)
    print(s_mean, s_std)
