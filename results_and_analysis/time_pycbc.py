from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pycbc
import sys
import time 
from tqdm import tqdm 
from pycbc import waveform


sns.set(style="ticks")
plt.rcParams['font.size']     = 18
plt.rcParams['font.family']   = 'Arial'
plt.rcParams['axes.titlesize']= 18
plt.rcParams['axes.labelsize']= 20
plt.rcParams['legend.fontsize']=16



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
    X_train = np.vstack([
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
        "X_train":    X_train,
        "y_amplitude": y_amplitude,
        "y_phase":     y_phase
    }
    return dataset_dict



n_workers  = 48
fs         = 4096
duration   = 2      
f_lower    = 50    
sample_sizes = [1, 10, 50, 100, 500, 1000]
approximants = [
    'SpinTaylorT1',
    'IMRPhenomPv2_NRTidal',
    'IMRPhenomPv2_NRTidalv2',
    'IMRPhenomXP_NRTidalv2'
]

def _gen_chunk(args):
    chunk_size, approx = args
    return generate_dataset_BNS(chunk_size, duration, fs, f_lower, approx)

if __name__ == "__main__":
    generation_times = {}
    for approx in approximants:
        generation_times[approx] = {}
        for total in sample_sizes:
            base = total // n_workers
            sizes = [base] * n_workers
            sizes[-1] += total - base * n_workers
            tasks = [(sz, approx) for sz in sizes]
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=n_workers) as exe:
                _ = list(exe.map(_gen_chunk, tasks))
            t1 = time.time()
            elapsed = t1 - t0
            generation_times[approx][total] = elapsed
            print(f"{approx:25s} | batch {total:5d} → {elapsed:.4f} s")

    save_dict = {
        f"{approx}_{batch}": generation_times[approx][batch]
        for approx in approximants
        for batch  in sample_sizes
    }
    np.savez_compressed("/path/to/your/data/pycbc_times_cpu.npz", **save_dict)

