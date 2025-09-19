import os
import numpy as np
import pandas as pd
import scipy.io
import pywt
from scipy.fft import rfft

RAW_MANIFEST = "data/manifest.csv"
OUT_FEATURES = "data/features.csv"
FIXED_LEN = 1024

def load_mat_signal(path):
    mat = scipy.io.loadmat(path)
    return mat['sig'].squeeze()

def pad_or_crop(signal, target_len=FIXED_LEN):
    if len(signal) >= target_len:
        return signal[:target_len]
    else:
        return np.pad(signal, (0, target_len-len(signal)), 'constant')

def extract_fft_features(signal):
    yf = np.abs(rfft(signal))
    feats = [np.mean(yf), np.std(yf), np.max(yf)]
    return feats

def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    feats = []
    for c in coeffs:
        feats.extend([np.mean(c), np.std(c)])
    return feats

def main():
    df = pd.read_csv(RAW_MANIFEST)
    rows = []

    for _, r in df.iterrows():
        path = r['file']

        # 🔧 Fix incorrect "../" paths automatically
        if path.startswith("../"):
            path = path.replace("../", "", 1)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        sig = load_mat_signal(path)
        sig = pad_or_crop(sig)

        row = []
        row.extend(extract_fft_features(sig))
        row.extend(extract_wavelet_features(sig))
        row.append(r['label'])
        rows.append(row)

    cols = ['fft_mean','fft_std','fft_max'] + \
           [f'wl{i}_{s}' for i in range(4) for s in ['mean','std']] + \
           ['label']

    pd.DataFrame(rows, columns=cols).to_csv(OUT_FEATURES, index=False)
    print(f"✅ Features saved to {OUT_FEATURES}")

if __name__ == "__main__":
    main()
