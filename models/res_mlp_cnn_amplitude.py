import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
import tensorflow_probability as tfp
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
from tensorflow.keras.callbacks import Callback
import tensorflow_addons as tfa 
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
tfb=tfp.bijectors
tfd=tfp.distributions



epochs=500
batch_size=128
dataset_path = '/path/to/your/data/BNS_train_set.npz'
dataset_dict = np.load(dataset_path)
X_train       = dataset_dict['X_train'].astype('float32')                 
y_amplitude       = dataset_dict['y_amplitude'].astype('float32')      

scaler_X = MinMaxScaler()
X_train_normalized = scaler_X.fit_transform(X_train)
scaler_amplitude = StandardScaler()
y_amplitude_normalized = scaler_amplitude.fit_transform(y_amplitude)


seq_len   = y_amplitude_normalized.shape[1]
param_dim = X_train_normalized.shape[1]

def residual_mlp_block(x, units: int, name_prefix: str):
    shortcut = x
    x = tfkl.Dense(units, activation='relu', name=f'{name_prefix}_dense1')(x)
    x = tfkl.Dense(units, activation=None,  name=f'{name_prefix}_dense2')(x)
    x = tfkl.Add(name=f'{name_prefix}_add')([shortcut, x])
    return tfkl.Activation('relu', name=f'{name_prefix}_relu')(x)

def residual_cnn_block(x, filters: int, name_prefix: str):
    in_channels = tf.keras.backend.int_shape(x)[-1]
    if in_channels != filters:
        shortcut = tfkl.Conv1D(filters, 1, padding='same',
                               name=f'{name_prefix}_proj_short')(x)
    else:
        shortcut = x
    x = tfkl.Conv1D(filters, 3, padding='same',
                    name=f'{name_prefix}_conv1')(x)
    x = tfa.layers.InstanceNormalization(name=f'{name_prefix}_in1')(x)
    x = tfkl.Activation('relu', name=f'{name_prefix}_relu1')(x)
    x = tfkl.Conv1D(filters, 3, padding='same',
                    name=f'{name_prefix}_conv2')(x)
    x = tfa.layers.InstanceNormalization(name=f'{name_prefix}_in2')(x)
    x = tfkl.Add(name=f'{name_prefix}_add')([shortcut, x])
    x = tfkl.Activation('relu', name=f'{name_prefix}_relu2')(x)

    return x


def build_waveform_net(label: str,
                       param_dim: int,
                       seq_len: int,
                       mlp_units: int = 1024,
                       mlp_blocks: int = 8,
                       cnn_filters: int = 32,
                       cnn_blocks: int = 3,
                       activation='tanh'):
    inp = tfkl.Input(shape=(param_dim,),
                     name=f'input_params_{label}')

    x = tfkl.Dense(mlp_units,
                   activation='relu',
                   name=f'{label}_mlp_input_dense')(inp)
    for i in range(mlp_blocks):
        x = residual_mlp_block(x,
                               mlp_units,
                               name_prefix=f'{label}_res_mlp{i+1}')
    x = tfkl.Dense(seq_len * cnn_filters,
                   activation='relu',
                   name=f'{label}_mlp_to_cnn_dense')(x)
    x = tfkl.Reshape((seq_len, cnn_filters),
                     name=f'{label}_reshape_to_cnn')(x)

    for j in range(cnn_blocks):
        x = residual_cnn_block(x,
                               cnn_filters,
                               name_prefix=f'{label}_res_cnn{j+1}')
    out = tfkl.Conv1D(1,
                      kernel_size=1,
                      activation=activation,
                      name=f'{label}_output_conv')(x)    
    out = tfkl.Lambda(lambda t: tf.squeeze(t, axis=-1),
                      name=f'{label}_squeeze')(out)      

    model = tfk.Model(inputs=inp,
                      outputs=out,
                      name=f'AI_{label}')
    return model

 


model = build_waveform_net(label='amplitude_res_mlp_cnn',param_dim=param_dim, seq_len=seq_len,mlp_units=1024,mlp_blocks=8,cnn_filters=32,cnn_blocks=3)
model.summary(line_length=120)
loss_fn   = tfk.losses.MeanSquaredError()
optimizer = tfk.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
reduce_lr = tfk.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            patience=7,
                                            factor=0.7,
                                            min_lr=1e-8,
                                            verbose=1)
early_stop = tfk.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=15,
                                         restore_best_weights=True,
                                         verbose=1)

callbacks_amplitude = [reduce_lr, early_stop]
history = model.fit(
    X_train_normalized,                 
    y_amplitude_normalized,            
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.10,
    callbacks=callbacks_amplitude,
    shuffle=True,  
)

model.save('/path/to/your/model/res_mlp_cnn_amplitude', save_format='tf')


hist = history.history
loss_keys = [k for k in hist.keys() if 'loss' in k]
if 'lr' in hist:
    loss_keys.append('lr')
losses = {k: np.array(hist[k]) for k in loss_keys}
print("Recorded keys:", list(losses.keys()))

np.save(
    '/path/to/your/data/res_mlp_cnn_losses_amplitude.npy',
    losses
)

