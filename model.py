import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    BatchNormalization,
    Add,
    Multiply,
    Conv2DTranspose
)
import numpy as np
import time
from tensorflow.keras.utils import plot_model


# ============================================================
# SeConv Block
# ============================================================

class SeConv_block(keras.layers.Layer):

    def __init__(
        self,
        kernel_size,
        input_channels,
        use_mask=True,
        use_selective=True,
        **kwargs
    ):

        super(SeConv_block, self).__init__()

        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.use_mask = use_mask
        self.use_selective = use_selective

    def build(self, input_shape):

        kernel_init = tf.ones_initializer()

        self.kernel = tf.Variable(
            name="kernel",
            initial_value=kernel_init(
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.input_channels,
                    1
                ),
                dtype='float32'
            ),
            trainable=True
        )

    def call(self, inputs):

        # ----------------------------------------------------
        # Binary Mask Guidance
        # ----------------------------------------------------

        if self.use_mask:

            M_hat = tf.math.not_equal(inputs, 0)
            M_hat = tf.cast(M_hat, tf.float32)

        else:

            M_hat = tf.ones_like(inputs)

        # ----------------------------------------------------
        # Convolution
        # ----------------------------------------------------

        conv_input = tf.nn.conv2d(
            inputs,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        conv_M_hat = tf.nn.conv2d(
            M_hat,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        # ----------------------------------------------------
        # Avoid division by zero
        # ----------------------------------------------------

        is_zero = tf.equal(conv_M_hat, 0)
        is_zero = tf.cast(is_zero, tf.float32)

        conv_M_hat = conv_M_hat + is_zero

        # ----------------------------------------------------
        # Weighted estimation
        # ----------------------------------------------------

        S = tf.divide(conv_input, conv_M_hat)

        # ----------------------------------------------------
        # Noise Map
        # ----------------------------------------------------

        M = 1 - M_hat

        # ----------------------------------------------------
        # Noise Recognition Matrix
        # ----------------------------------------------------

        kernel_ones = np.ones(
            (
                self.kernel_size,
                self.kernel_size,
                self.input_channels,
                1
            )
        )

        kernel_ones = tf.constant(kernel_ones, dtype=tf.float32)

        R = tf.nn.conv2d(
            M_hat,
            kernel_ones,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        R = tf.greater_equal(
            R,
            tf.constant(self.kernel_size - 2, dtype=tf.float32)
        )

        R = tf.cast(R, tf.float32)

        # ----------------------------------------------------
        # Selective Restoration
        # ----------------------------------------------------

        if self.use_selective:

            y = tf.multiply(tf.multiply(S, M), R) + inputs

        else:

            y = S

        return y

    def get_config(self):

        config = super(SeConv_block, self).get_config()

        config.update({
            "kernel_size": self.kernel_size,
            "use_mask": self.use_mask,
            "use_selective": self.use_selective
        })

        return config


# ============================================================
# NASED-Net Model
# ============================================================

def SeConvNet(

    num_SeConv_block=7,
    filters=64,
    image_channels=1,

    # Ablation switches
    use_mask=True,
    use_selective=True,
    adaptive_kernel=True,

    # Depth analysis
    encoder_depth=2

):

    layer_count = 0

    inputs = Input(
        shape=(None, None, image_channels),
        name='input'
    )

    x = inputs

    # ========================================================
    # SeConv Blocks
    # ========================================================

    for i in range(num_SeConv_block):

        layer_count += 1

        # Adaptive kernel sizes
        if adaptive_kernel:

            kernel_size = 2 * layer_count + 1

        else:

            kernel_size = 3

        x = SeConv_block(
            kernel_size=kernel_size,
            input_channels=image_channels,
            use_mask=use_mask,
            use_selective=use_selective,
            name='SeConv_block_' + str(layer_count)
        )(x)

    # ========================================================
    # Encoder
    # ========================================================

    # Encoder 1
    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_initializer='Orthogonal',
        name='Encoder_Conv1'
    )(x)

    x = BatchNormalization(name='Encoder_BN1')(x)

    x = Activation('relu', name='Encoder_ReLU1')(x)

    # Encoder 2
    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_initializer='Orthogonal',
        name='Encoder_Conv2'
    )(x)

    x = BatchNormalization(name='Encoder_BN2')(x)

    x = Activation('relu', name='Encoder_ReLU2')(x)

    # ========================================================
    # Decoder
    # ========================================================

    # Decoder 1
    x = Conv2DTranspose(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_initializer='Orthogonal',
        name='Decoder_Conv0'
    )(x)

    x = BatchNormalization(name='Decoder_BN0')(x)

    x = Activation('relu', name='Decoder_ReLU0')(x)

    # Decoder 2
    x = Conv2DTranspose(
        filters=image_channels,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_initializer='Orthogonal',
        name='Decoder_Conv1'
    )(x)

    # ========================================================
    # Final Mask-Guided Restoration
    # ========================================================

    if use_mask:

        x = Multiply(name='Final_Multiply')([
            x,
            tf.cast(tf.equal(inputs, 0), tf.float32)
        ])

        outputs = Add(name='Final_Add')([x, inputs])

    else:

        outputs = x

    # ========================================================
    # Create Model
    # ========================================================

    model = keras.models.Model(
        inputs=inputs,
        outputs=outputs,
        name='NASED_Net'
    )

    return model


# ============================================================
# MODEL COMPLEXITY FUNCTION
# ============================================================

def compute_model_complexity(model):

    print("\n================ MODEL COMPLEXITY ================\n")

    # --------------------------------------------------------
    # Parameter Count
    # --------------------------------------------------------

    total_params = model.count_params()

    trainable_params = np.sum([
        np.prod(v.shape)
        for v in model.trainable_weights
    ])

    non_trainable_params = np.sum([
        np.prod(v.shape)
        for v in model.non_trainable_weights
    ])

    print(f"Total Parameters       : {total_params:,}")
    print(f"Trainable Parameters   : {trainable_params:,}")
    print(f"Non-Trainable Params   : {non_trainable_params:,}")

    # ========================================================
    # Inference Time
    # ========================================================

    dummy_input = tf.random.normal((1, 40, 40, 1))

    # Warmup
    _ = model(dummy_input)

    start = time.time()

    for _ in range(50):

        _ = model(dummy_input)

    end = time.time()

    avg_inference_time = (end - start) / 50

    print(f"\nAverage Inference Time : {avg_inference_time:.6f} seconds")


# ============================================================
# Different Ablation Models
# ============================================================

# 1. Full Model
full_model = SeConvNet()

# 2. Without Binary Mask Guidance
no_mask_model = SeConvNet(
    use_mask=True
)

# 3. Fixed Kernel Model
fixed_kernel_model = SeConvNet(
    adaptive_kernel=True
)

# 4. Without Selective Restoration
no_selective_model = SeConvNet(
    use_selective=True
)

# 5. Deep Encoder-Decoder
deep_model = SeConvNet(
    encoder_depth=2
)

# ============================================================
# Print Summary
# ============================================================

print("\n================ FULL MODEL ================\n")

full_model.summary()

print("\nTotal Parameters:",
      full_model.count_params())

# ============================================================
# Compute Complexity
# ============================================================

compute_model_complexity(full_model)

# ============================================================
# Save Architecture Figure
# ============================================================

plot_model(
    full_model,
    to_file='NASED_Net_Flowchart.png',
    show_shapes=True,
    show_layer_names=True
)
