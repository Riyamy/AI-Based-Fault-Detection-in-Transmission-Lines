import scipy.io
import matplotlib.pyplot as plt
import glob

# Pick a few sample .mat files
files = glob.glob("data/raw/*.mat")[:4]  # first 4 signals

for f in files:
    mat = scipy.io.loadmat(f)
    sig = mat['sig'].squeeze()
    t = mat['t'].squeeze()

    plt.figure()
    plt.plot(t, sig)
    plt.title(f"Signal: {f.split('/')[-1]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
