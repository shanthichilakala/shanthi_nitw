import numpy as np
import cv2
import glob
import os


# ============================================================
# Settings
# ============================================================

patch_size = 40
stride = 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


# ============================================================
# Data Augmentation
# ============================================================

def data_aug(img, mode=0):

    if mode == 0:
        return img

    elif mode == 1:
        return np.flipud(img)

    elif mode == 2:
        return np.rot90(img)

    elif mode == 3:
        return np.flipud(np.rot90(img))

    elif mode == 4:
        return np.rot90(img, k=2)

    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))

    elif mode == 6:
        return np.rot90(img, k=3)

    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


# ============================================================
# Generate Patches
# ============================================================

def gen_patches(file_name):

    img = cv2.imread(file_name, -1)

    # Check image exists
    if img is None:

        print("Could not read :", file_name)

        return []

    h, w = img.shape[:2]

    patches = []

    # --------------------------------------------------------
    # Multi-scale patches
    # --------------------------------------------------------

    for s in scales:

        h_scaled = int(h * s)
        w_scaled = int(w * s)

        img_scaled = cv2.resize(

            img,
            (w_scaled, h_scaled),
            interpolation=cv2.INTER_CUBIC

        )

        # ----------------------------------------------------
        # Sliding window patches
        # ----------------------------------------------------

        for i in range(
            0,
            h_scaled - patch_size + 1,
            stride
        ):

            for j in range(
                0,
                w_scaled - patch_size + 1,
                stride
            ):

                x = img_scaled[
                    i:i + patch_size,
                    j:j + patch_size
                ]

                # --------------------------------------------
                # Augmentation
                # --------------------------------------------

                for k in range(aug_times):

                    mode = np.random.randint(0, 8)

                    x_aug = data_aug(
                        x,
                        mode=mode
                    )

                    patches.append(x_aug)

    return patches


# ============================================================
# Dataset Generator
# ============================================================

def data_gen(

    data_dir='home/ubuntu22/Shanthi_paper1_files/1_paper1_revision/data/Train/Gray',
    verbose=False

):

    ext = ['png', 'jpg', 'jpeg', 'bmp', 'gif']

    file_list = []

    # --------------------------------------------------------
    # Read image files
    # --------------------------------------------------------

    for e in ext:

        file_list.extend(

            glob.glob(

                os.path.join(
                    data_dir,
                    '*.' + e
                )

            )

        )

    # --------------------------------------------------------
    # Debug Information
    # --------------------------------------------------------

    print("\n===================================")
    print("Dataset Directory :", data_dir)
    print("Total Images Found :", len(file_list))
    print("===================================\n")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if len(file_list) == 0:

        raise ValueError(

            "\nNo images found in:\n"
            + data_dir

        )

    # --------------------------------------------------------
    # Generate patches
    # --------------------------------------------------------

    data = []

    for i in range(len(file_list)):

        patch = gen_patches(file_list[i])

        if len(patch) > 0:

            data.extend(patch)

        if verbose:

            print(

                str(i + 1)
                + '/'
                + str(len(file_list))
                + ' processed'

            )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    data = np.array(data)

    print("\nTotal Patches Extracted :", len(data))

    # --------------------------------------------------------
    # Check patch extraction
    # --------------------------------------------------------

    if len(data) == 0:

        raise ValueError(

            "\nNo patches extracted.\n"
            "Check image size and patch_size."

        )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    if data.ndim == 3:

        # grayscale

        data = data.reshape(

            (
                data.shape[0],
                data.shape[1],
                data.shape[2],
                1
            )

        )

    elif data.ndim == 4:

        # color

        pass

    else:

        raise ValueError(

            "\nUnexpected data shape : "
            + str(data.shape)

        )

    # --------------------------------------------------------
    # Remove incomplete batch
    # --------------------------------------------------------

    discard_n = len(data) % batch_size

    if discard_n > 0:

        data = data[:-discard_n]

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\n===================================")
    print("Training Dataset Prepared")
    print("Dataset Shape :", data.shape)
    print("Patch Size    :", patch_size)
    print("Stride        :", stride)
    print("Scales        :", scales)
    print("===================================\n")

    return data
