import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pycbc
import sys
import time 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from pycbc.waveform.utils import amplitude_from_polarizations, phase_from_polarizations,frequency_from_polarizations
from scipy.interpolate import interp1d
from pycbc import waveform
from pycbc.waveform import get_td_waveform ,get_fd_waveform

Kpc = 1e3 * 3.0857e16    
Mpc = 1e6 * 3.0857e16   
Gpc = 1e9 * 3.0857e16    
solar_mass = 1.98847e30 



num_samples = int(1e5)
fs  = 4096
times = 2  #s
f_lower=50
def cal_generate_BNS_waveform(approximant='IMRPhenomXP_NRTidalv2', **kwargs):
    waveform_args = {"approximant": approximant}
    for key, value in kwargs.items():
        if value is not None:  
            waveform_args[key] = value
    hp, hc = waveform.get_td_waveform(**waveform_args)
    return hp, hc


def cal_sample_mass(num_samples, m1_min, m1_max, m2_min=None, m2_max=None):
    if m2_min is None:
        m2_min = m1_min
    if m2_max is None:
        m2_max = m1_max

    masse1 = np.random.uniform(m1_min, m1_max, num_samples)
    masse2 = np.random.uniform(m2_min, m2_max, num_samples)
    masses1 = np.maximum(masse1, masse2)
    masses2 = np.minimum(masse1, masse2)

    if num_samples == 1:
        return masses1[0], masses2[0]  
    return masses1, masses2  

def cal_sample_tidal_deformability(num_samples, lambda1_min, lambda1_max):
    lambda1 = np.random.uniform(lambda1_min, lambda1_max, num_samples)
    if num_samples == 1:
        return lambda1[0] 
    return lambda1 

def cal_sample_spin(num_samples, max_spin):
    if num_samples == 1:
        v = np.random.normal(size=3)
        v /= np.linalg.norm(v)
        r = np.random.uniform(0, max_spin)
        return v[0] * r, v[1] * r, v[2] * r

    vecs = np.random.normal(size=(num_samples, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)    
    radii = np.random.uniform(0, max_spin, size=(num_samples, 1)) 
    spins = vecs * radii                                     
    return spins[:, 0], spins[:, 1], spins[:, 2]

def cal_amplitude_from_hphc_pycbc(hp, hc):
    amplitude = waveform.utils.amplitude_from_polarizations(hp, hc)
    return amplitude

def cal_phase_from_hphc_pycbc(hp, hc):
    phase = waveform.utils.phase_from_polarizations(hp, hc)
    phase=phase-phase[0]
    return phase

def cal_pad_array(arr,target_length):
    if len(arr) < target_length:
        result = np.pad(arr, (0, target_length - len(arr)), 'constant', constant_values=0)
    else:
        result = arr[-target_length:]
    return result


def generate_dataset_BNS(num_samples, times, sample_frequency, f_lower=50,approximant='IMRPhenomXP_NRTidalv2'):
    masses1, masses2 = cal_sample_mass(num_samples, 1, 3)
    max_spin = 0.5
    spin1xs, spin1ys, spin1zs = cal_sample_spin(num_samples, max_spin)
    spin2xs, spin2ys, spin2zs = cal_sample_spin(num_samples, max_spin)
    lambda1s = cal_sample_tidal_deformability(num_samples, 0, 500)
    lambda2s = cal_sample_tidal_deformability(num_samples, 0, 500)
    X_test = np.vstack([
        masses1, masses2,
        lambda1s, lambda2s,
        spin1xs, spin1ys, spin1zs,
        spin2xs, spin2ys, spin2zs
    ]).T.astype(np.float32)
    pad_len = int(times * sample_frequency)
    y_amplitude = np.zeros((num_samples, pad_len), dtype=np.float32)
    y_phase     = np.zeros((num_samples, pad_len), dtype=np.float32)

    for i in tqdm(range(num_samples)):
        hp, hc = cal_generate_BNS_waveform(
            approximant=approximant,
            mass1   = masses1[i],
            mass2   = masses2[i],
            delta_t = 1 / sample_frequency,
            f_lower = f_lower,
            spin1x  = spin1xs[i],
            spin2x  = spin2xs[i],
            spin1y  = spin1ys[i],
            spin2y  = spin2ys[i],
            spin1z  = spin1zs[i],
            spin2z  = spin2zs[i],
            lambda1 = lambda1s[i],
            lambda2 = lambda2s[i],
        )

        hp = hp.trim_zeros()
        hc = hc.trim_zeros()

        amplitude = np.array(cal_amplitude_from_hphc_pycbc(hp, hc), dtype=np.float32)
        phase     = np.array(cal_phase_from_hphc_pycbc(hp, hc), dtype=np.float32)

        amplitude = cal_pad_array(amplitude, pad_len)
        phase     = cal_pad_array(phase, pad_len)
        phase = phase - phase[0]
        y_amplitude[i, :] = amplitude
        y_phase[i, :]     = phase

    dataset_dict = {
        "X_test":    X_test,
        "y_amplitude": y_amplitude,
        "y_phase":     y_phase
    }
    return dataset_dict

def _generate_chunk(chunk_size):
    return generate_dataset_BNS(num_samples, times, fs,f_lower)


if __name__ == "__main__":
    n_workers = 32
    base = num_samples // n_workers
    sizes = [base] * n_workers
    sizes[-1] += num_samples - base * n_workers
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        results = list(exe.map(_generate_chunk, sizes))

    X_parts     = [d["X_test"]    for d in results]
    amp_parts   = [d["y_amplitude"] for d in results]
    phase_parts = [d["y_phase"]     for d in results]

    X_test     = np.concatenate(X_parts,   axis=0)
    y_amplitude = np.concatenate(amp_parts, axis=0)
    y_phase     = np.concatenate(phase_parts, axis=0)
    save_path = "/path/to/your/data/BNS_test_set.npz"
    np.savez(
        save_path,
        X_test=X_test,
        y_amplitude=y_amplitude,
        y_phase=y_phase
    )
















