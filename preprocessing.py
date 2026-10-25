import numpy as np
import scipy.signal as signal

def extract_csi(csi_data):
    """Mengekstrak komponen Amplitude dan Phase (jika input kompleks) dari matriks CSI."""
    if np.iscomplexobj(csi_data):
        return np.abs(csi_data), np.angle(csi_data)
    return csi_data.astype(np.float32), None

def apply_butterworth_lowpass(data, wn=0.18, order=4, axis=1):
    """Menerapkan Low-pass Butterworth Filter menggunakan frekuensi ternormalisasi Wn."""
    b, a = signal.butter(order, wn, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=axis).astype(np.float32)

def sanitize_phase(raw_phase):
    """Menerapkan linear phase unwrapping dan transformasi linier antar-subcarrier."""
    unwrapped = np.unwrap(raw_phase, axis=-1)
    F = unwrapped.shape[-1]
    a = (unwrapped[..., -1] - unwrapped[..., 0]) / (F - 1)
    b = np.mean(unwrapped, axis=-1)
    idx = np.arange(F)
    return unwrapped - (a[..., np.newaxis] * idx + b[..., np.newaxis])

def compute_stft_spectrogram(sig_1d, fs=1.0, nperseg=32, noverlap=24):
    """Mengubah sinyal CSI 1D menjadi spektrogram STFT setelah menghapus DC offset."""
    sig_zero_mean = sig_1d - np.mean(sig_1d)
    f, t, Zxx = signal.stft(sig_zero_mean, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)