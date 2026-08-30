"""
tf.data pipeline for training the TFLite compression model.

Compatible with the OpenImages directory structure used by the original
PyTorch training (data/openimages/), and with a flat directory of images such
as data/coffee/train/. Both work because the globs enumerate directory depths
explicitly — see _image_patterns; list_files cannot glob recursively.
"""

import tensorflow as tf

CROP_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE

IMAGE_EXTS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

# tf.data.Dataset.list_files cannot glob recursively: "**" matches exactly one
# directory level, the same as "*". A lone "{root}/**/*.jpg" therefore finds
# images one level below root and nowhere else, silently missing images that
# sit directly in root — which is how pointing --dataset_path at a flat folder
# fails with "No files matched pattern". Enumerate the depths instead.
_GLOB_DEPTHS = ("", "*/", "*/*/")


def _image_patterns(root):
    """Glob patterns covering images in root and up to two levels below."""
    root = str(root).rstrip("/")
    return [f"{root}/{depth}*.{ext}"
            for depth in _GLOB_DEPTHS
            for ext in IMAGE_EXTS]


def _load_and_preprocess(path, crop_size=CROP_SIZE, training=True):
    """
    Load an image file and return a cropped, normalised (B=1) tensor.

    Args:
        path:      string tensor — path to JPEG or PNG
        crop_size: int — output spatial size
        training:  if True, apply random crop + flip; else centre-crop

    Returns:
        float32 tensor (crop_size, crop_size, 3) in [0, 1]
    """
    raw = tf.io.read_file(path)
    # Try JPEG first; fall back to PNG
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([None, None, 3])

    if training:
        # Random scale between 0.75–0.95 of the shorter side, then crop
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        min_dim = tf.minimum(h, w)
        scale = tf.random.uniform([], 0.75, 0.95)
        new_min = tf.cast(tf.cast(min_dim, tf.float32) * scale, tf.int32)
        new_min = tf.maximum(new_min, crop_size)
        new_h = tf.cast(tf.cast(h, tf.float32) * tf.cast(new_min, tf.float32)
                        / tf.cast(min_dim, tf.float32), tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * tf.cast(new_min, tf.float32)
                        / tf.cast(min_dim, tf.float32), tf.int32)
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.random_crop(image, [crop_size, crop_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        # Centre crop
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        offset_h = (h - crop_size) // 2
        offset_w = (w - crop_size) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, crop_size, crop_size
        )

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def get_dataset(root, training=True, batch_size=8, crop_size=CROP_SIZE,
                shuffle_buffer=10000):
    """
    Build a tf.data.Dataset from a directory of images.

    Args:
        root:           Path to dataset directory (e.g. 'data/openimages').
                        Scanned recursively for *.jpg / *.jpeg / *.png.
        training:       bool — applies augmentation and shuffling when True.
        batch_size:     int
        crop_size:      int — spatial crop (must be 256 for this model)
        shuffle_buffer: int — shuffle buffer size

    Returns:
        Batched tf.data.Dataset of float32 tensors (B, 256, 256, 3).
    """
    files = tf.data.Dataset.list_files(
        _image_patterns(root), shuffle=training
    )

    ds = files.map(
        lambda p: _load_and_preprocess(p, crop_size, training),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.shuffle(shuffle_buffer)

    ds = (ds
          .batch(batch_size, drop_remainder=True)
          .prefetch(AUTOTUNE))

    return ds


def get_calibration_dataset(root, n_samples=200, crop_size=CROP_SIZE):
    """
    Return a small fixed dataset for INT8 quantization calibration.

    Yields single-image batches (1, 256, 256, 3) as numpy arrays.
    """
    files = tf.data.Dataset.list_files(_image_patterns(root), shuffle=False)
    files = files.take(n_samples)

    ds = files.map(
        lambda p: _load_and_preprocess(p, crop_size, training=False),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(1)

    def gen():
        for batch in ds:
            yield [batch.numpy()]

    return gen


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/openimages"
    ds = get_dataset(root, training=True, batch_size=4)
    for batch in ds.take(1):
        print(f"Batch shape: {batch.shape}  min={batch.numpy().min():.3f}  max={batch.numpy().max():.3f}")
    print("Data pipeline OK")
