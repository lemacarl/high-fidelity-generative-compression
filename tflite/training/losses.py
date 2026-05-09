"""
Loss functions for TFLite compression model training.

Total loss = weighted_rate + k_M * MSE + k_P * (1 - MS-SSIM)

GAN loss is handled directly in trainer.py (not needed here for Phase 1).
"""

import tensorflow as tf

# Default weights matching original HIFIC defaults
K_M = 0.075 * (2 ** -5)   # MSE weight (~0.00234)
K_P = 1.0                  # Perceptual (MS-SSIM) weight
LAMBDA_A = 2.0             # Rate penalty when above target (low regime)
LAMBDA_B = 2 ** -4         # Rate penalty when below target (0.0625)
TARGET_BPP = 0.14          # Target bits-per-pixel (low regime)


def rate_loss(bpp, target_bpp=TARGET_BPP, lambda_a=LAMBDA_A, lambda_b=LAMBDA_B):
    """
    Adaptive rate penalty — heavier above target, lighter below.

    Args:
        bpp:        scalar tensor — current bits per pixel
        target_bpp: float — target operating point
        lambda_a:   float — penalty above target (default 2.0 for low regime)
        lambda_b:   float — penalty below target (default 0.0625)

    Returns:
        scalar rate loss tensor
    """
    weight = tf.where(bpp > target_bpp, lambda_a, lambda_b)
    return weight * bpp


def distortion_loss(x, x_hat, k_m=K_M):
    """
    Pixel-space MSE loss (computed in [0, 255] range, same as original).

    Args:
        x, x_hat: (B, H, W, 3) float32 in [0, 1]

    Returns:
        scalar MSE loss
    """
    return k_m * tf.reduce_mean(tf.square((x - x_hat) * 255.0))


def ms_ssim_loss(x, x_hat, k_p=K_P, max_val=255.0):
    """
    MS-SSIM perceptual loss: k_P * (1 - MS-SSIM).

    Operates in [0, 255] space. Returns 0 for perfect reconstruction.

    Args:
        x, x_hat: (B, H, W, 3) float32 in [0, 1]

    Returns:
        scalar MS-SSIM loss
    """
    ms_ssim_val = tf.image.ssim_multiscale(
        x * max_val, x_hat * max_val, max_val=max_val,
        filter_size=11, filter_sigma=1.5,
        k1=0.01, k2=0.03,
    )
    # Guard against NaN/Inf that can arise with low-variance patches;
    # treat them as no perceptual signal (0 loss contribution).
    ms_ssim_val = tf.where(tf.math.is_finite(ms_ssim_val),
                           ms_ssim_val, tf.zeros_like(ms_ssim_val))
    ms_ssim_val = tf.clip_by_value(ms_ssim_val, 0.0, 1.0)
    return k_p * tf.reduce_mean(1.0 - ms_ssim_val)


def total_compression_loss(x, x_hat, hyper_bpp, latent_bpp,
                           target_bpp=TARGET_BPP,
                           lambda_a=LAMBDA_A, lambda_b=LAMBDA_B,
                           k_m=K_M, k_p=K_P):
    """
    Phase 1 (compression-only) total loss.

    Returns:
        total_loss, rate, distortion, perceptual  — all scalar tensors
    """
    total_bpp = hyper_bpp + latent_bpp
    r_loss = rate_loss(total_bpp, target_bpp, lambda_a, lambda_b)
    d_loss = distortion_loss(x, x_hat, k_m)
    p_loss = ms_ssim_loss(x, x_hat, k_p)
    total = r_loss + d_loss + p_loss
    return total, r_loss, d_loss, p_loss


def gan_generator_loss(d_fake, loss_type="non_saturating"):
    """
    GAN generator loss (Phase 2).

    d_fake: discriminator output for generated images, shape (B, P, P, 1)
    """
    if loss_type == "non_saturating":
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_fake), logits=d_fake
            )
        )
    elif loss_type == "least_squares":
        return 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    raise ValueError(f"Unknown GAN loss type: {loss_type}")


def gan_discriminator_loss(d_real, d_fake, loss_type="non_saturating"):
    """
    GAN discriminator loss (Phase 2).
    """
    if loss_type == "non_saturating":
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(d_real), logits=d_real
        )
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_fake), logits=d_fake
        )
        return tf.reduce_mean(real_loss + fake_loss)
    elif loss_type == "least_squares":
        return 0.5 * (tf.reduce_mean(tf.square(d_real - 1.0))
                      + tf.reduce_mean(tf.square(d_fake)))
    raise ValueError(f"Unknown GAN loss type: {loss_type}")
