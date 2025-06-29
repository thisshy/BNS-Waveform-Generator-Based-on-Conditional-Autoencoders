import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("❌ Failed to set memory growth:", e)

def cal_hphc_from_amp_ph(amplitude, phase):
    hp = amplitude * np.cos(phase)
    hc = amplitude * np.sin(phase)
    return hp, hc

dataset_path = '/path/to/your/data/BNS_train_set.npz'
dataset_dict = np.load(dataset_path)
X_train       = dataset_dict['X_train'].astype('float32')                 
y_amplitude       = dataset_dict['y_amplitude'].astype('float32')      
y_phase       = dataset_dict['y_phase '].astype('float32') 

scaler_X = MinMaxScaler();       scaler_X.fit(X_train)
scaler_y_phase = StandardScaler();     scaler_y_phase.fit(y_phase)
scaler_y_amplitude = StandardScaler(); scaler_y_amplitude.fit(y_amplitude)

cAE_encoder_phase_conditional = tf.keras.models.load_model(
    '/path/to/your/model/cAE_encoder_phase_conditional'
)
cAE_decoder_phase = tf.keras.models.load_model(
    '/path/to/your/model/cAE_decoder_phase'
)

cAE_encoder_amplitude_conditional = tf.keras.models.load_model(
    '/path/to/your/model/cAE_encoder_amplitude_conditional'
)
cAE_decoder_amplitude = tf.keras.models.load_model(
    '/path/to/your/model/cAE_decoder_amplitude'
)

def predict_waveform_from_sample(x_sample):
    x_sample = np.array(x_sample).reshape(1, -1)
    x_scaled = scaler_X.transform(x_sample)

    phase_z = cAE_encoder_phase_conditional(x_scaled)
    phase_AI = cAE_decoder_phase(phase_z).numpy().ravel()

    amp_z = cAE_encoder_amplitude_conditional(x_scaled)
    amplitude_AI = cAE_decoder_amplitude(amp_z).numpy().ravel()

    phase_AI = scaler_y_phase.inverse_transform(phase_AI.reshape(1, -1)).ravel()
    amplitude_AI = scaler_y_amplitude.inverse_transform(amplitude_AI.reshape(1, -1)).ravel()

    hp_pred, hc_pred = cal_hphc_from_amp_ph(amplitude_AI, phase_AI)
    return hp_pred, hc_pred
















