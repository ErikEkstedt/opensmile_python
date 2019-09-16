import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from subprocess import check_output, run, Popen, PIPE, STDOUT, DEVNULL, call
from os.path import join, expanduser, basename
from os import system

BIN_PATH = "./opensmile/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract"
CONF_PATH = "./config/gemaps_50ms/GeMAPSv01a.conf"


def plot_gemap(data, figsize=(12, 12)):
    num_plots = 18
    j = 1
    fig = plt.figure(figsize=figsize)
    for k, v in data.items():
        if k == "name":
            continue
        if not k == "frameTime":
            plt.subplot(num_plots, 1, j)
            if v.ndim == 1:
                plt.plot(v, label=k)
            else:
                plt.plot(v[0], alpha=0.5, label=f"{k} 0")
                plt.plot(v[1], alpha=0.5, label=f"{k} 1")
            if not j == num_plots:
                plt.xticks([])
            plt.legend(loc="upper right")
            j += 1
    plt.tight_layout()
    plt.pause(0.01)
    return fig


def plot_gemap_no_freq(data, figsize=(12, 12)):
    num_plots = 10
    j = 1
    fig = plt.figure(figsize=figsize)
    for k, v in data.items():
        if k == "name":
            continue
        if not k.lower().startswith("f"):
            plt.subplot(num_plots, 1, j)
            if v.ndim == 1:
                plt.plot(v, label=k)
            else:
                plt.plot(v[0], alpha=0.5, label=f"{k} 0")
                plt.plot(v[1], alpha=0.5, label=f"{k} 1")
            if not j == num_plots:
                plt.xticks([])
            plt.legend(loc="upper right")
            j += 1
    plt.tight_layout()
    plt.pause(0.01)
    return fig


def read_csv(path, delimiter=","):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            data.append(row)
    return data


def get_info_sox(fpath):
    out = (
        check_output(f"sox --i {fpath}", shell=True).decode("utf-8").strip().split("\n")
    )
    info = {}
    for line in out:
        l = [f for f in line.split(" ") if not f == "" and not f == ":"]
        typ = l[0].lower()
        if typ.startswith("channels"):
            info[typ] = int(l[1])
        elif typ == "sample":
            if l[1].lower() == "rate":
                info["sample_rate"] = int(l[2])
            else:
                info["sample_encoding"] = l[2] + " " + l[3]
        elif typ == "precision":
            info["precision"] = l[1]
        elif typ == "bit":
            info["bit_rate"] = l[2]
        elif typ == "file":
            info["filesize"] = l[2]
        elif line.lower().startswith("duration"):
            duration = l[1].split(":")
            hh, mm, ss = duration
            info["duration"] = int(hh) * 60 * 60 + int(mm) * 60 + float(ss)
    return info


def split_sox(wav_path, tmp_path="/tmp"):
    wav_name = basename(wav_path)
    to_path = join(tmp_path, wav_name.strip(".wav"))
    to_path_0 = to_path + "_0.wav"
    to_path_1 = to_path + "_1.wav"
    cmd = ["sox", wav_path, "-b", "16", to_path_0, "remix", str(1)]
    run(cmd)
    cmd = ["sox", wav_path, "-b", "16", to_path_1, "remix", str(2)]
    run(cmd)
    return to_path_0, to_path_1


def extract(wav_path, out_path, out_func, conf_path, bin_path):
    cmd = [
        bin_path,
        "-C",
        conf_path,
        "-l",
        "0",
        "-I",
        wav_path,
        "-D",
        out_path,
        "-csvoutput",
        out_func,
    ]
    call(cmd, stdin=PIPE, stdout=DEVNULL, stderr=STDOUT)


def read_gemaps(feat_path, func_path):
    renaming = {
        "Loudness_sma3": "loudness",
        "alphaRatio_sma3": "alphaRatio",
        "hammarbergIndex_sma3": "hammarberg_index",
        "slope0-500_sma3": "slope0_500",
        "slope500-1500_sma3": "slope500_1500",
        "F0semitoneFrom27.5Hz_sma3nz": "pitch",
        "jitterLocal_sma3nz": "jitter",
        "shimmerLocaldB_sma3nz": "shimmer",
        "HNRdBACF_sma3nz": "HNR",
        "logRelF0-H1-H2_sma3nz": "harmonic_diff_H1_H2",
        "logRelF0-H1-A3_sma3nz": "harmonic_diff_H1_A3",
        "F1bandwidth_sma3nz": "F1_bandwidth",
        "F1frequency_sma3nz": "F1_freq",
        "F2frequency_sma3nz": "F2_freq",
        "F3frequency_sma3nz": "F3_freq",
        "F1amplitudeLogRelF0_sma3nz": "F1_rel",
        "F2amplitudeLogRelF0_sma3nz": "F2_rel",
        "F3amplitudeLogRelF0_sma3nz": "F3_rel",
    }

    gc = read_csv(feat_path, ";")
    feats = gc[0][2:]
    features = {}
    for f in feats:
        features[renaming[f]] = []

    for frame in gc[1:]:
        for k, v in zip(feats, frame[2:]):
            features[renaming[k]].append(float(v))

    for k, v in features.items():
        features[k] = np.array(v, dtype=np.float32)

    gf = read_csv(func_path, ";")
    funcs = gf[0][2:]
    values = gf[1][2:]
    fun = {}
    for func, val in zip(funcs, values):
        for name, new_name in renaming.items():
            if func.startswith(name):
                new_func = new_name
                break
        ren = func.replace(name, new_name)
        fun[ren] = float(val)

    return features, fun


class GemapExtractor(object):
    def __init__(self, conf_path=CONF_PATH, bin_path=BIN_PATH, tmp_path="/tmp"):
        self.conf_path = conf_path
        self.bin_path = bin_path
        self.tmp_path = tmp_path

    def __call__(self, wav_path):
        info = get_info_sox(wav_path)
        out_path = join(self.tmp_path, "gemap_features.csv")
        out_func = join(self.tmp_path, "gemap_funcs.csv")
        if info["channels"] == 2:
            ch0_path, ch1_path = split_sox(wav_path)
            features, functionals = [], []
            for wpath in [ch0_path, ch1_path]:
                _ = system(f"rm {out_path}")
                _ = system(f"rm {out_func}")
                extract(wpath, out_path, out_func, self.conf_path, self.bin_path)
                feats, funcs = read_gemaps(out_path, out_func)
                features.append(feats)
                functionals.append(funcs)

            feats = {}
            for k in features[0]:
                feats[k] = np.stack((features[0][k], features[1][k]))

            funcs = {}
            for k in functionals[0]:
                funcs[k] = np.stack((functionals[0][k], functionals[1][k]))
        else:
            _ = system(f"rm {out_path}")
            _ = system(f"rm {out_func}")
            extract(wav_path, out_path, out_func, self.conf_path, self.bin_path)
            feats, funcs = read_gemaps(out_path, out_func)
        return feats, funcs


# used in another repo
def compare():
    import parselmouth
    from turntaking.dataprocessing.Process import extract_melspectrogram
    from librosa import power_to_db

    # Compare frames
    snds = parselmouth.Sound(wav_path).extract_all_channels()
    melspec = []
    for snd in snds:
        melspec.append(
            extract_melspectrogram(
                snd,
                80,
                0.05,
                0.05,
                snd.sampling_frequency,
                fmax=snd.sampling_frequency // 2,
            ).T
        )
    melspec = np.stack(melspec)
    print("melspec: ", melspec.shape)
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(melspec[0].T, aspect="auto", origin="bottom")
    plt.subplot(2, 1, 2)
    plt.imshow(melspec[1].T, aspect="auto", origin="bottom")
    plt.tight_layout()
    plt.pause(0.01)
    return fig, melspec


if __name__ == "__main__":
    # Original
    # bin_path = "./opensmile/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract"
    # conf_path = "opensmile/opensmile-2.3.0/config/gemaps/GeMAPSv01a.conf"

    Gemap = GemapExtractor()

    # SWB
    wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw2001.wav")
    # wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw2300.wav")
    # wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw2301.wav")
    # wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw2289.wav")
    # wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw2752.wav")
    # wav_path = join(expanduser("~"), "SpeechCorpus/switchboard/data/audio/sw3647.wav")

    gemaps, funcs = Gemap(wav_path)

    for k, v in gemaps.items():
        print(f"{k}: {v.shape}")
    # for k, v in funcs.items():
    #     print(f"{k}: {v.shape}")
    fig = plot_gemap(gemaps)

    fig, melspec = compare()
