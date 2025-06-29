import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
import tensorflow_probability as tfp
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import Callback

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
tfb=tfp.bijectors
tfd=tfp.distributions


def cal_normalize_data(x_list, y=None, x_scalers=None, y_scaler=None, print_name_list=None):
    if not isinstance(x_list, (list, tuple)):
        x_list = [x_list]
    n = len(x_list)
    if x_scalers is None:
        x_scalers = [StandardScaler() for _ in range(n)]
    elif not isinstance(x_scalers, (list, tuple)):
        x_scalers = [x_scalers] * n
    if print_name_list is None:
        print_name_list = [f"x{i}" for i in range(n)]
    elif isinstance(print_name_list, str):
        print_name_list = [print_name_list] * n
    x_normalized_list = []
    for xi, scaler, name in zip(x_list, x_scalers, print_name_list):
        arr = np.asarray(xi)
        x_norm = scaler.fit_transform(arr).astype("float32")
        print(f"{name}_scaled shape:", x_norm.shape)
        x_normalized_list.append(x_norm)
    y_normalized = None
    if y is not None:
        arr_y = np.asarray(y)
        y_normalized = y_scaler.fit_transform(arr_y).astype("float32")
        print("y_scaled shape:", y_normalized.shape)
    return x_normalized_list, y_normalized, x_scalers, y_scaler


def cal_prepare_datasets(
    x_norm,
    y_norm,
    save_dir,
    train_fraction,
    val_fraction,
    N_blocks,
    batch_size,
    as_dict=False,
    input_names=None,
):
    if isinstance(x_norm, (list, tuple)):
        flat = []
        for item in x_norm:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        x_norm = flat
    def _to32(arr):
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    if isinstance(x_norm, np.ndarray):
        x_norm = _to32(x_norm)
    else:
        x_norm = [_to32(arr) for arr in x_norm]
    y_norm = _to32(y_norm)
    if isinstance(x_norm, np.ndarray):
        x_list = [x_norm]
    elif isinstance(x_norm, (list, tuple)):
        x_list = list(x_norm)
    else:
        raise TypeError("x_norm must be ndarray or list/tuple of ndarray")
    N_set = {arr.shape[0] for arr in x_list}
    if len(N_set) != 1:
        raise ValueError(f"Inconsistent sample counts across inputs: {N_set}")
    N_total = x_list[0].shape[0]
    para_lens = [arr.shape[1] for arr in x_list]
    num_inputs = len(x_list)
    y_dim = y_norm.shape[1]
    if as_dict:
        if input_names is None:
            input_names = [f"input_{i}" for i in range(num_inputs)]
        if len(input_names) != num_inputs:
            raise ValueError("input_names length must equal number of inputs")
    else:
        input_names = None
    block_size = N_total // N_blocks
    os.makedirs(save_dir, exist_ok=True)
    expected = [f"X{j}_block{i}.npy" for j in range(num_inputs) for i in range(N_blocks)]
    expected += [f"y_block{i}.npy" for i in range(N_blocks)]
    missing = [fn for fn in expected if not os.path.exists(os.path.join(save_dir, fn))]
    if missing:
        print(f"❗ Detected {len(missing)} missing block files → regenerating …")
        for i in range(N_blocks):
            need = any(name in missing for name in 
                       [f"X{j}_block{i}.npy" for j in range(num_inputs)] + [f"y_block{i}.npy"])
            if not need:
                continue
            s = i * block_size
            e = (i + 1) * block_size if i < N_blocks - 1 else N_total
            for j, arr in enumerate(x_list):
                np.save(os.path.join(save_dir, f"X{j}_block{i}.npy"), arr[s:e])
            np.save(os.path.join(save_dir, f"y_block{i}.npy"), y_norm[s:e])
        print("✅ Missing blocks regenerated")
    else:
        print(f"✅ '{save_dir}' already has complete {N_blocks} blocks, skipping split")
    if abs(train_fraction + val_fraction - 1) > 1e-6:
        raise ValueError("train_fraction + val_fraction must sum to 1")
    num_train = int(N_blocks * train_fraction)
    train_ids = list(range(num_train))
    val_ids = list(range(num_train, N_blocks))
    train_samples = block_size * len(train_ids)
    val_samples = N_total - train_samples
    train_steps = train_samples // batch_size
    val_steps = val_samples // batch_size
    def make_ds(ids, shuffle_blocks, shuffle_samples):
        ds_list = []
        for j in range(num_inputs):
            paths = [os.path.join(save_dir, f"X{j}_block{i}.npy") for i in ids]
            ds = tf.data.Dataset.from_tensor_slices(paths)
            if shuffle_blocks:
                ds = ds.shuffle(len(ids))
            ds_list.append(ds)
        paths_y = [os.path.join(save_dir, f"y_block{i}.npy") for i in ids]
        ds_y = tf.data.Dataset.from_tensor_slices(paths_y)

        ds = tf.data.Dataset.zip(tuple(ds_list + [ds_y]))

        def load_block(*fps):
            *x_fps, y_fp = fps
            Xs = []
            for xp, D in zip(x_fps, para_lens):
                arr = tf.numpy_function(lambda fn: np.load(fn.decode()).astype(np.float32),
                                        [xp], tf.float32)
                arr.set_shape([None, D])
                Xs.append(arr)
            Yb = tf.numpy_function(lambda fn: np.load(fn.decode()).astype(np.float32),
                                   [y_fp], tf.float32)
            Yb.set_shape([None, y_dim])
            if as_dict:
                feat = {name: Xs[i] for i, name in enumerate(input_names)}
            else:
                feat = tuple(Xs) if len(Xs) > 1 else Xs[0]
            return feat, Yb

        ds = ds.map(load_block, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.unbatch()
        if shuffle_samples:
            ds = ds.shuffle(buffer_size=block_size*2)
        return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    train_ds = make_ds(train_ids, shuffle_blocks=True, shuffle_samples=True).repeat()
    val_ds = make_ds(val_ids, shuffle_blocks=False, shuffle_samples=False)
    return train_steps, val_steps, train_ds, val_ds


def cal_transformer_block(x, attn_heads, key_dim, ff_dim, attn_dropout, attn_dropout_post, ff_dropout, name_prefix):
    attn_output = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=key_dim, dropout=attn_dropout, name=f'{name_prefix}_mha')(x, x)
    attn_output = layers.Dropout(attn_dropout_post, name=f'{name_prefix}_attn_dropout')(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_attn_layernorm')(x + attn_output)
    ff_output = layers.Dense(ff_dim, activation='relu', name=f'{name_prefix}_ff_dense1')(x)
    ff_output = layers.Dense(x.shape[-1], name=f'{name_prefix}_ff_dense2')(ff_output)
    ff_output = layers.Dropout(ff_dropout, name=f'{name_prefix}_ff_dropout')(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ff_layernorm')(x + ff_output)
    return x


initial_lr=1e-4
epoch=500

dataset_path = '/path/to/your/data/BNS_train_set.npz'
dataset_dict = np.load(dataset_path)
X_train       = dataset_dict['X_train'].astype('float32')                 
y_phase       = dataset_dict['y_phase'].astype('float32')         

data_len = y_phase.shape[1]    
para_len = X_train.shape[1]   
N_total  = X_train.shape[0]   

X_train_normalized, y_phase_normalized, scaler_X, scaler_y = cal_normalize_data(
    x_list=X_train,
    y=y_phase,
    x_scalers=MinMaxScaler(),
    y_scaler=StandardScaler(),
    print_name_list='phase'
)

train_steps, val_steps, train_dataset, val_dataset =cal_prepare_datasets(
    x_norm=[y_phase_normalized,X_train_normalized],
    y_norm=y_phase_normalized,
    save_dir="/path/to/your/data/blocks_phase",
    train_fraction=0.9,
    val_fraction=0.1,
    N_blocks=10,
    batch_size=128
)



def resnet_block(x, filters, kernel_size, strides=1, name_prefix=None):
    shortcut = x
    y = tfkl.Conv1D(
        filters, 
        kernel_size=kernel_size, 
        strides=strides, 
        padding='same',
        use_bias=False, 
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv1' if name_prefix else None
    )(x)
    y = tfkl.BatchNormalization(name=f'{name_prefix}_bn1' if name_prefix else None)(y)
    y = tfkl.Activation('relu', name=f'{name_prefix}_act1' if name_prefix else None)(y)
    y = tfkl.Conv1D(
        filters, 
        kernel_size=kernel_size, 
        strides=1, 
        padding='same',
        use_bias=False, 
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv2' if name_prefix else None
    )(y)
    y = tfkl.BatchNormalization(name=f'{name_prefix}_bn2' if name_prefix else None)(y)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tfkl.Conv1D(
            filters, 
            kernel_size=1, 
            strides=strides, 
            padding='same',
            use_bias=False, 
            kernel_initializer='he_normal',
            name=f'{name_prefix}_shortcut_conv' if name_prefix else None
        )(shortcut)
        shortcut = tfkl.BatchNormalization(
            name=f'{name_prefix}_shortcut_bn' if name_prefix else None
        )(shortcut)
    out = tfkl.Add(name=f'{name_prefix}_add' if name_prefix else None)([shortcut, y])
    out = tfkl.Activation('relu', name=f'{name_prefix}_act2' if name_prefix else None)(out)
    return out


def cAE_model(data_len, para_len, label):

    latent_dim = 300
    attn_heads = 6
    key_dim = 16
    ff_dim = 64
    attn_dropout = 0.1
    attn_dropout_post = 0.1
    ff_dropout = 0.1

    ###encoder parameters
    input_params = Input(shape=(para_len,), name=f'input_params_{label}')
    x_params = layers.Dense(50, activation='relu')(input_params)
    x_params = layers.Dense(50, activation='relu')(x_params)
    x_params = layers.Dense(50, activation='relu')(x_params)
    x_params_seq = layers.Reshape((1, 50),
                                 name=f'encoder_params_reshape_{label}')(x_params)
    x_params_trans = cal_transformer_block(
        x_params_seq, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'encoder_params_transformer1_{label}')
    x_params_trans = cal_transformer_block(
        x_params_trans, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'encoder_params_transformer2_{label}')
    x_params_flat = layers.Flatten(
        name=f'encoder_params_flatten_{label}')(x_params_trans)
    mean_1 = layers.Dense(latent_dim,
                          name=f'encoder_params_mean_{label}')(x_params_flat)

    ###encoder phase
    input_strain = Input(shape=(data_len,),
                         name=f'input_strain_{label}')
    x = layers.Reshape((data_len, 1),
                       name=f'reshape_input_{label}')(input_strain)

    x = layers.Conv1D(16, 3, strides=2, padding='same',
                      name=f'{label}_downsample_conv1')(x)
    x = layers.ReLU()(x)
    x = resnet_block(x, 16)

    x = layers.Conv1D(32, 3, strides=2, padding='same',
                      name=f'{label}_downsample_conv2')(x)
    x = layers.ReLU()(x)
    x = resnet_block(x, 32)

    x = layers.Conv1D(64, 3, strides=2, padding='same',
                      name=f'{label}_downsample_conv3')(x)
    x = layers.ReLU()(x)
    x = resnet_block(x, 64)

    x = cal_transformer_block(
        x, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'encoder_phase_transformer1_{label}')
    x = cal_transformer_block(
        x, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'encoder_phase_transformer2_{label}')

    x_enc = layers.GlobalAveragePooling1D(
        name=f'global_pool_{label}')(x)
    mean = layers.Dense(latent_dim,
                        name=f'encoder_mean_{label}')(x_enc)

    ###decoder
    decoder_input = Input(shape=(latent_dim,),
                          name=f'decoder_input_{label}')
    reduced_len = data_len // 8
    x_dec = layers.Dense(reduced_len * 64,
                         name=f'{label}_dec_dense')(decoder_input)
    x_dec = layers.Reshape((reduced_len, 64),
                           name=f'{label}_dec_reshape')(x_dec)

    x_dec = cal_transformer_block(
        x_dec, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'decoder_transformer1_{label}')
    x_dec = cal_transformer_block(
        x_dec, attn_heads, key_dim, ff_dim,
        attn_dropout, attn_dropout_post, ff_dropout,
        name_prefix=f'decoder_transformer2_{label}')

    x_dec = layers.Conv1DTranspose(32, 3, strides=2, padding='same',
                                   name=f'{label}_upsample_conv1')(x_dec)
    x_dec = layers.ReLU()(x_dec)
    x_dec = resnet_block(x_dec, 32)

    x_dec = layers.Conv1DTranspose(16, 3, strides=2, padding='same',
                                   name=f'{label}_upsample_conv2')(x_dec)
    x_dec = layers.ReLU()(x_dec)
    x_dec = resnet_block(x_dec, 16)

    x_dec = layers.Conv1DTranspose(8, 3, strides=2, padding='same',
                                   name=f'{label}_upsample_conv3')(x_dec)
    x_dec = layers.ReLU()(x_dec)
    x_dec = resnet_block(x_dec, 8)

    decoded = layers.Conv1D(1, 1, activation='linear', padding='same',
                            name=f'{label}_decoder_final_conv')(x_dec)
    decoded_output = layers.Reshape((data_len,),
                                    name=f'decoder_output_{label}')(decoded)

    encoder_params_phase = Model(
        inputs=input_params,
        outputs=mean_1,
        name=f'encoder_params_{label}'
    )
    encoder_phase = Model(
        inputs=input_strain,
        outputs=mean,
        name=f'encoder_{label}'
    )
    decoder_phase = Model(
        inputs=decoder_input,
        outputs=decoded_output,
        name=f'decoder_{label}'
    )

    encoded_p = encoder_phase(input_strain)
    decoded_p = decoder_phase(encoded_p)
    autoencoder_phase = Model(
        inputs=[input_strain, input_params],
        outputs=decoded_p,
        name=f'autoencoder_{label}'
    )

    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mae(input_strain, decoded_p)
    )
    latent_loss = tf.reduce_mean(
        tf.keras.losses.mse(mean, mean_1)
    )
    total_loss = reconstruction_loss + latent_loss

    autoencoder_phase.add_loss(total_loss)
    autoencoder_phase.add_metric(
        reconstruction_loss,
        name=f'reconstruction_loss_{label}',
        aggregation='mean'
    )
    autoencoder_phase.add_metric(
        latent_loss,
        name=f'latent_loss_{label}',
        aggregation='mean'
    )
    autoencoder_phase.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    )

    return autoencoder_phase, encoder_params_phase, encoder_phase, decoder_phase



autoencoder_phase, encoder_params_phase, encoder_phase, decoder_phase = cAE_model(data_len, para_len, 'phase')
autoencoder_phase.summary()
autoencoder_phase.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr))


checkpoint_filepath = ('/path/to/your/data/phase_checkpoint.h5')
if os.path.exists(checkpoint_filepath):
    print(f"Found checkpoint at {checkpoint_filepath}. Loading weights...")
    autoencoder_phase.load_weights(checkpoint_filepath)
else:
    print("No checkpoint found. Training from scratch.")


reduce_lr = tfk.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=7, factor=0.7, min_lr=1e-7
)
es = tfk.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    verbose=0,
    save_weights_only=True
)

callbacks_phase = [reduce_lr, es, checkpoint_cb]

history = autoencoder_phase.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epoch,                       
    verbose=1,
    steps_per_epoch=train_steps,   
    validation_steps=val_steps,     
    callbacks=callbacks_phase
)


encoder_phase.save(
    '/path/to/your/model/cAE_encoder_phase_conditional',
    save_format='tf'
)
decoder_phase.save(
    '/path/to/your/model/cAE_decoder_phase',
    save_format='tf'
)




hist = history.history
loss_keys = [k for k in hist.keys() if 'loss' in k]
if 'lr' in hist:
    loss_keys.append('lr')
losses = {k: np.array(hist[k]) for k in loss_keys}
print("Recorded keys:", list(losses.keys()))

np.save(
    '/path/to/your/data/cAE_losses_phase.npy',
    losses
)






















