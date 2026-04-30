import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
import csv
import os
from datetime import datetime
from scipy.io.wavfile import write

# =======================
# INPUT UTILISATEUR
# =======================
nom = input("Nom de l'essai : ")
num_exp = input("Numéro de l'essai : ")

plt.close("all")

# =======================
# PARAMÈTRES
# =======================
FS = 48000
DURATION = 2
CHANNELS = 2

ACC_CHANNEL = 0
MIC_CHANNEL = 1

SENS_ACCEL = 0.01      # V/g
SENS_MIC = 0.05        # V/Pa
AUDIO_FULL_SCALE_V = 1.0

# =======================
# ACQUISITION
# =======================
print("Acquisition en cours...")

sd.default.device = 26  # Remplacez par l'index de votre interface audio (voir sd.query_devices())
data = sd.rec(
    int(DURATION * FS),
    samplerate=FS,
    channels=CHANNELS,
    dtype='float32'
)
sd.wait()
print("Acquisition terminée")

# =======================
# SAUVEGARDE AUDIO
# =======================

# Normalisation pour éviter clipping
audio_norm = data / np.max(np.abs(data))

# Conversion en int16 (format WAV standard)
audio_int16 = (audio_norm * 32767).astype(np.int16)



# =======================
# EXTRACTION
# =======================
acc_signal = data[:, ACC_CHANNEL]
mic_signal = data[:, MIC_CHANNEL]

# =======================
# FILTRE PASSE-HAUT
# =======================
def highpass(sig, fs, cutoff=1):
    b, a = butter(2, cutoff/(fs/2), btype='high')
    return filtfilt(b, a, sig)

acc_signal = highpass(acc_signal, FS)
mic_signal = highpass(mic_signal, FS)

# =======================
# CONVERSIONS
# =======================
acc_voltage = acc_signal * AUDIO_FULL_SCALE_V
mic_voltage = mic_signal * AUDIO_FULL_SCALE_V

acc_g = acc_voltage / SENS_ACCEL
pressure = mic_voltage / SENS_MIC

# =======================
# NORMALISATION
# =======================
def normalize(sig):
    max_val = np.max(np.abs(sig))
    return sig if max_val == 0 else sig / max_val

acc_g = normalize(acc_g)
pressure = normalize(pressure)

# =======================
# TEMPS
# =======================
N = len(acc_g)
t = np.arange(N) / FS

# =======================
# FFT
# =======================
window = np.hanning(N)

acc_fft = np.abs(np.fft.rfft(acc_g * window)) / N
mic_fft = np.abs(np.fft.rfft(pressure * window)) / N

freq = np.fft.rfftfreq(N, 1/FS)

# =======================
# LISSAGE
# =======================
sigma = 10
acc_fft = gaussian_filter1d(acc_fft, sigma)
mic_fft = gaussian_filter1d(mic_fft, sigma)

# =======================
# PASSAGE dB
# =======================
acc_fft_db = 20 * np.log10(acc_fft + 1e-12)
mic_fft_db = 20 * np.log10(mic_fft + 1e-12)

# =======================
# DÉTECTION DES PICS
# =======================
min_height_db = -80
min_distance = 20
prominence = 5

# Accéléromètre
peaks_acc, _ = find_peaks(
    acc_fft_db,
    height=min_height_db,
    distance=min_distance,
    prominence=prominence
)

sorted_acc = sorted(peaks_acc, key=lambda p: acc_fft_db[p], reverse=True)

print("\nPics Accéléromètre :")
for p in sorted_acc[:10]:
    if freq[p] < 3000:
        print(f"{freq[p]:.1f} Hz | {acc_fft_db[p]:.1f} dB")

# Micro
peaks_mic, _ = find_peaks(
    mic_fft_db,
    height=min_height_db,
    distance=min_distance,
    prominence=prominence
)

sorted_mic = sorted(peaks_mic, key=lambda p: mic_fft_db[p], reverse=True)

print("\nPics Microphone :")
for p in sorted_mic[:10]:
    if freq[p] < 3000:
        print(f"{freq[p]:.1f} Hz | {mic_fft_db[p]:.1f} dB")

# =======================
# ZOOM TEMPOREL
# =======================
delta_t = 0.05
peak_index = np.argmax(np.abs(acc_g))
delta_n = int(delta_t * FS)

start = max(0, peak_index - delta_n)
end = min(N, peak_index + delta_n)

t_zoom = t[start:end]
acc_zoom = acc_g[start:end]
mic_zoom = pressure[start:end]

# =======================
# GRAPHES
# =======================
fig, ax = plt.subplots(2, 1, figsize=(12,7))

# TEMPS
ax[0].plot(t_zoom, acc_zoom, label="Accéléromètre", color="blue")
ax[0].plot(t_zoom, mic_zoom, label="Micro", color="red")
ax[0].set_title("Signal temporel")
ax[0].grid()
ax[0].legend()

# FFT LOG
ax[1].plot(freq, acc_fft_db, label="Accéléromètre", color="blue")
ax[1].plot(freq, mic_fft_db, label="Micro", color="red")

# AJOUT DES PICS
ax[1].plot(freq[sorted_acc[:10]], acc_fft_db[sorted_acc[:10]], "bx")
ax[1].plot(freq[sorted_mic[:10]], mic_fft_db[sorted_mic[:10]], "rx")

ax[1].set_xlim(10, 2000)
ax[1].set_title("FFT (log)")
ax[1].grid(which="both")
ax[1].legend()


plt.tight_layout()

plt.show()

if input("Garder ?") == 'Y':
    # Sauvegarde
    filename = f"{nom}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    fig.savefig(filename, dpi=300)
    audio_filename = f"{nom}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
    write(audio_filename, FS, audio_int16)

    print(f"Fichier audio sauvegardé : {audio_filename}")

    # =======================
    # SAUVEGARDE CSV
    # =======================
    filename = "pics.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "nom",
                "timestamp",
                "num_exp",
                "frequence_hz",
                "amplitude_db"
            ])

        now = datetime.now().isoformat()

        # ACC
        for p in sorted_acc[:10]:
            if freq[p] < 3000:
                writer.writerow([now, nom, num_exp, "acc", freq[p], acc_fft_db[p]])

        # MIC
        for p in sorted_mic[:10]:
            if freq[p] < 3000:
                writer.writerow([now, nom, num_exp, "mic", freq[p], mic_fft_db[p]])

    print("Pics FFT enregistrés dans pics.csv")