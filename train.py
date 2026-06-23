# ============================================================
# UPDATED TRAIN FILE WITH SAVED PLOTS
# Reviewer Required Additions Included
#
# SAVES:
# 1. Training Loss Curve
# 2. Runtime vs Noise Density Plot
# 3. Runtime CSV
# 4. Training History CSV
# 5. Hyperparameter Log
#
# SAVE LOCATION:
# /home/ubuntu22/Shanthi_paper1
#
# FIGURE QUALITY:
# dpi = 600
#
# ORIGINAL MODEL/TRAINING SETTINGS NOT CHANGED
# ============================================================

import argparse
import numpy as np
from tensorflow import keras
from tensorflow.math import reduce_sum, square
import tensorflow as tf
import os
import time

# ============================================================
# FIX Qt/XCB ERROR
# ============================================================

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

from model import SeConvNet
from SPN import SPN
from data_generator import data_gen


# ============================================================
# MAIN SAVE DIRECTORY
# ============================================================

MAIN_SAVE_DIR = "/home/ubuntu22/Shanthi_paper1"

os.makedirs(MAIN_SAVE_DIR, exist_ok=True)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    '--noise_density',
    default=0.9,
    type=float
)

parser.add_argument(
    '--image_channels',
    default=1,
    type=int
)

parser.add_argument(
    '--epoch',
    dest='epoch',
    type=int,
    default=50
)

parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=128
)

parser.add_argument(
    '--lr',
    dest='lr',
    type=float,
    default=1e-3
)

# ============================================================
# UPDATED TRAIN DIRECTORY
# ============================================================

parser.add_argument(
    '--train_dir',
    default='/home/ubuntu22/Shanthi_paper1_files/1_paper1_revision/data/Train/Gray',
    type=str,
    help='path of train data'
)

parser.add_argument(
    '--steps',
    dest='steps',
    type=int,
    default=2000
)

args = parser.parse_args(args=[])


# ============================================================
# Color Mode
# ============================================================

color_mode = 'Gray' if args.image_channels == 1 else 'Color'

print("\nColor Mode :", color_mode)


# ============================================================
# Save Directory
# ============================================================

save_dir = os.path.join(
    MAIN_SAVE_DIR,
    'weights',
    color_mode,
    'SeConvNet_' + str(int(100 * args.noise_density))
)

os.makedirs(save_dir, exist_ok=True)


# ============================================================
# Create Model
# ============================================================

model = SeConvNet(
    image_channels=args.image_channels,
    encoder_depth=2
)

model.summary()


# ============================================================
# MODEL COMPLEXITY ANALYSIS
# ============================================================

print("\n================================================")
print("MODEL COMPLEXITY ANALYSIS")
print("================================================")

total_params = model.count_params()

trainable_params = np.sum(
    [np.prod(v.shape) for v in model.trainable_weights]
)

non_trainable_params = np.sum(
    [np.prod(v.shape) for v in model.non_trainable_weights]
)

print(f"\nTotal Parameters        : {total_params:,}")
print(f"Trainable Parameters    : {trainable_params:,}")
print(f"Non-Trainable Parameters: {non_trainable_params:,}")


# ============================================================
# Memory Footprint
# ============================================================

memory_mb = (total_params * 4) / (1024**2)

print(f"\nApprox Model Size       : {memory_mb:.2f} MB")


# ============================================================
# FLOPs Calculation
# ============================================================

try:

    concrete_func = tf.function(
        lambda inputs: model(inputs)
    ).get_concrete_function(
        tf.TensorSpec(
            [1, 40, 40, args.image_channels],
            tf.float32
        )
    )

    frozen_func = concrete_func.graph

    run_meta = tf.compat.v1.RunMetadata()

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func,
        run_meta=run_meta,
        cmd='op',
        options=opts
    )

    print(f"\nApprox FLOPs            : {flops.total_float_ops:,}")

except:

    print("\nFLOPs Calculation Failed")


# ============================================================
# GPU Inference Time
# ============================================================

dummy_input = tf.random.normal(
    (1, 40, 40, args.image_channels)
)

_ = model(dummy_input, training=False)

num_runs = 100

start_time = time.time()

for _ in range(num_runs):

    _ = model(dummy_input, training=False)

end_time = time.time()

gpu_inference = (
    (end_time - start_time) / num_runs
)

print(f"\nGPU Inference Time      : {gpu_inference:.6f} sec")


# ============================================================
# CPU Inference Time
# ============================================================

with tf.device('/CPU:0'):

    start_time = time.time()

    for _ in range(20):

        _ = model(dummy_input, training=False)

    end_time = time.time()

cpu_inference = (
    (end_time - start_time) / 20
)

print(f"CPU Inference Time      : {cpu_inference:.6f} sec")

print("================================================\n")


# ============================================================
# Runtime vs Noise Density
# ============================================================

print("\n================================================")
print("RUNTIME VS NOISE DENSITY")
print("================================================")

noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

runtime_results = []

sample = np.random.rand(
    1,
    40,
    40,
    args.image_channels
).astype(np.float32)

for nd in noise_levels:

    noisy_sample = SPN(
        sample,
        density=nd
    )

    start = time.time()

    _ = model(noisy_sample, training=False)

    runtime = time.time() - start

    runtime_results.append(runtime)

    print(
        f"Noise Density {nd:.1f} "
        f"-> Runtime: {runtime:.6f} sec"
    )


# ============================================================
# Save Runtime CSV
# ============================================================

runtime_df = pd.DataFrame({

    'Noise Density': noise_levels,
    'Runtime (sec)': runtime_results

})

runtime_df.to_csv(

    os.path.join(
        MAIN_SAVE_DIR,
        'runtime_analysis.csv'
    ),

    index=False
)


# ============================================================
# Save Runtime Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(
    noise_levels,
    runtime_results,
    marker='o',
    linewidth=2
)

plt.xlabel('Noise Density')
plt.ylabel('Runtime (sec)')
plt.title('Runtime vs Noise Density')

plt.grid(True)

plt.savefig(

    os.path.join(
        MAIN_SAVE_DIR,
        'runtime_vs_noise_density.png'
    ),

    dpi=600,
    bbox_inches='tight'
)

plt.close()

print("\nRuntime plot saved.")


# ============================================================
# Data Generator
# ============================================================

def train_datagen(
    epoch_iter=2000,
    epoch_num=5,
    batch_size=args.batch_size,
    data_dir=args.train_dir
):

    while(True):

        n_count = 0

        if n_count == 0:

            xs = data_gen(data_dir)

            indices = list(range(xs.shape[0]))

            n_count = 1

        for _ in range(epoch_num):

            np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):

                batch_x = xs[indices[i:i+batch_size]]

                batch_x = batch_x.astype('float32') / 255.0

                batch_y = SPN(
                    batch_x,
                    args.noise_density
                )

                batch_y[batch_y == 1] = 0.

                yield batch_y, batch_x


# ============================================================
# Loss Function
# ============================================================

def sum_squared_error(y_true, y_pred):

    return reduce_sum(
        square(y_pred - y_true)
    ) / 2


# ============================================================
# Compile Model
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=args.lr
    ),

    loss=sum_squared_error
)


# ============================================================
# Hyperparameter Logging
# ============================================================

hyper_file = open(

    os.path.join(
        MAIN_SAVE_DIR,
        'hyperparameters.txt'
    ),

    'w'
)

hyper_file.write(f"Epochs           : {args.epoch}\n")
hyper_file.write(f"Batch Size       : {args.batch_size}\n")
hyper_file.write(f"Learning Rate    : {args.lr}\n")
hyper_file.write(f"Noise Density    : {args.noise_density}\n")
hyper_file.write(f"Optimizer        : Adam\n")
hyper_file.write(f"Loss Function    : SSE\n")

hyper_file.close()


# ============================================================
# Learning Rate Scheduler
# ============================================================

def scheduler(epoch):

    epochs = args.epoch

    initial_lr = args.lr

    if epoch <= int(0.7 * epochs):

        lr = initial_lr

    else:

        lr = initial_lr / 10

    print(
        'current learning rate is %1.8f'
        % lr
    )

    return lr


LearningRate_Scheduler = keras.callbacks.LearningRateScheduler(
    scheduler
)


# ============================================================
# Model Checkpoint
# ============================================================

model_checkpoint = keras.callbacks.ModelCheckpoint(

    os.path.join(
        save_dir,
        'model_{epoch:03d}.weights.h5'
    ),

    verbose=1,

    save_best_only=False,

    save_weights_only=True
)


# ============================================================
# CSV Logger
# ============================================================

csv_logger = keras.callbacks.CSVLogger(

    os.path.join(
        MAIN_SAVE_DIR,
        'training.log'
    ),

    separator=",",

    append=True
)


# ============================================================
# Training Time Callback
# ============================================================

class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):

        self.train_start = time.time()

        self.losses = []

        print("\n================================================")
        print("Training Started")
        print("================================================\n")

    def on_epoch_begin(self, epoch, logs=None):

        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):

        epoch_time = (
            time.time() - self.epoch_start
        )

        self.losses.append(logs['loss'])

        print(
            f"\nEpoch {epoch+1} Training Time: "
            f"{epoch_time:.2f} seconds"
        )

        print(
            f"Training Loss: "
            f"{logs['loss']:.6f}"
        )

    def on_train_end(self, logs=None):

        total_training_time = (
            time.time() - self.train_start
        )

        print("\n================================================")
        print(
            f"Total Training Time: "
            f"{total_training_time:.2f} seconds"
        )
        print("================================================\n")

        # ====================================================
        # SAVE TRAINING LOSS CURVE
        # ====================================================

        plt.figure(figsize=(8,6))

        plt.plot(
            self.losses,
            linewidth=2
        )

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')

        plt.grid(True)

        plt.savefig(

            os.path.join(
                MAIN_SAVE_DIR,
                'training_loss_curve.png'
            ),

            dpi=600,
            bbox_inches='tight'
        )

        plt.close()

        print(
            "Training loss curve saved."
        )


time_callback = TimeHistory()


# ============================================================
# Train Model
# ============================================================

history = model.fit(

    train_datagen(
        batch_size=args.batch_size
    ),

    steps_per_epoch=args.steps,

    epochs=args.epoch,

    verbose=1,

    callbacks=[

        model_checkpoint,

        csv_logger,

        LearningRate_Scheduler,

        time_callback
    ]
)


# ============================================================
# Save Final Training History
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(

    os.path.join(
        MAIN_SAVE_DIR,
        'training_history.csv'
    ),

    index=False
)


print("\n================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================================")

print("\nAll files saved in:")
print(MAIN_SAVE_DIR)
