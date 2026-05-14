"""
Training loop for TFLite generative compression model.

Usage:
    # Phase 1 – compression-only
    python -m tflite.training.trainer \
        --dataset_path data/openimages \
        --regime low \
        --batch_size 8 \
        --n_steps 500000 \
        --checkpoint_dir experiments/tflite_low/

    # Phase 2 – GAN fine-tuning (warm-start from Phase 1)
    python -m tflite.training.trainer \
        --dataset_path data/openimages \
        --regime low \
        --model_type compression_gan \
        --warmstart \
        --checkpoint experiments/tflite_low/ckpt-500000 \
        --n_steps 200000 \
        --checkpoint_dir experiments/tflite_low_gan/
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import time

import numpy as np
import tensorflow as tf

from tflite.model.compression_model import CompressionModel
from tflite.training.data_pipeline import get_dataset
from tflite.training import losses as loss_fn

# -------------------------------------------------------------------
# PatchGAN discriminator (Phase 2 only — not exported to TFLite)
# -------------------------------------------------------------------

def build_discriminator(image_shape=(256, 256, 3), latent_channels=96):
    """
    Full-capacity PatchGAN discriminator matching original HIFIC (~5 M params).

    Matches src/network/discriminator.py translated to TF/Keras:
      - Latent context path: Conv2D(C→12) + bilinear UpSampling2D(16×) → 256×256
      - Concatenated input: image (3ch) + context (12ch) = 15 channels
      - Strided conv tower: 15→64→128→256→512 with SpectralNormalization
      - Output: patch logit map (B, H', W', 1)

    Spectral normalization stabilises GAN training and is the primary
    difference that lifts discriminator quality over the previous design.

    Args:
        image_shape:     (H, W, C) of input images — default (256, 256, 3)
        latent_channels: Must match encoder output depth — default 96
    """
    image_in  = tf.keras.Input(shape=image_shape,              name="disc_image")
    latent_in = tf.keras.Input(shape=(16, 16, latent_channels), name="disc_latent")

    # ── Context path: latents → 12 channels → 256×256 ──────────────────────
    # Mirrors: self.context_conv + self.context_upsample in discriminator.py
    ctx = tf.keras.layers.Conv2D(
        12, 3, padding="same", use_bias=True, name="disc_ctx_conv"
    )(latent_in)
    ctx = tf.keras.layers.LeakyReLU(0.2, name="disc_ctx_lrelu")(ctx)
    ctx = tf.keras.layers.UpSampling2D(
        size=(16, 16), interpolation="bilinear", name="disc_ctx_up"
    )(ctx)   # (B, 16, 16, 12) → (B, 256, 256, 12)

    # ── Concatenate image + context: 3 + 12 = 15 channels ──────────────────
    x = tf.keras.layers.Concatenate(name="disc_cat")([image_in, ctx])

    # ── Strided conv tower with SpectralNormalization ───────────────────────
    # Mirrors: conv1…conv4 with spectral_norm in discriminator.py
    filters = [64, 128, 256, 512]
    for i, f in enumerate(filters):
        x = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(
                f, 4, strides=2, padding="same",
                use_bias=False, name=f"disc_conv{i}"
            ),
            name=f"disc_sn{i}"
        )(x)
        x = tf.keras.layers.LeakyReLU(0.2, name=f"disc_lrelu{i}")(x)

    # ── Patch logit output ──────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(1, 1, padding="same", name="disc_out")(x)

    return tf.keras.Model(
        inputs=[image_in, latent_in], outputs=x, name="discriminator"
    )


# -------------------------------------------------------------------
# Regime configurations
# -------------------------------------------------------------------

REGIME_CONFIG = {
    "low":  dict(target_bpp=0.14, lambda_a=2.0),
    "med":  dict(target_bpp=0.30, lambda_a=1.0),
    "high": dict(target_bpp=0.45, lambda_a=0.5),
}


# -------------------------------------------------------------------
# Training step functions
# -------------------------------------------------------------------

@tf.function
def compression_train_step(x, model, amort_opt, entropy_opt,
                            target_bpp, lambda_a):
    with tf.GradientTape(persistent=True) as tape:
        x_hat, hyper_bpp, latent_bpp = model(x, training=True)
        total, r, d, p = loss_fn.total_compression_loss(
            x, x_hat, hyper_bpp, latent_bpp,
            target_bpp=target_bpp, lambda_a=lambda_a,
        )

    amort_vars, entropy_vars = model.get_variable_groups()

    amort_grads = tape.gradient(total, amort_vars)
    entropy_grads = tape.gradient(total, entropy_vars)

    # Replace NaN/Inf gradients with zeros so one bad batch cannot
    # corrupt the Adam optimizer's moment state.
    amort_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in amort_grads
    ]
    entropy_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in entropy_grads
    ]

    amort_grads, _ = tf.clip_by_global_norm(amort_grads, 5.0)
    entropy_grads, _ = tf.clip_by_global_norm(entropy_grads, 5.0)

    amort_opt.apply_gradients(zip(amort_grads, amort_vars))
    entropy_opt.apply_gradients(zip(entropy_grads, entropy_vars))

    total_bpp = hyper_bpp + latent_bpp
    return total, r, d, p, total_bpp, x_hat


@tf.function
def generator_train_step(x, model, discriminator, amort_opt, entropy_opt,
                         target_bpp, lambda_a, beta, gan_loss_type):
    with tf.GradientTape(persistent=True) as tape:
        # return_latents=True so y_hat can be forwarded to the discriminator
        x_hat, hyper_bpp, latent_bpp, y_hat = model(
            x, training=True, return_latents=True
        )
        total, r, d, p = loss_fn.total_compression_loss(
            x, x_hat, hyper_bpp, latent_bpp,
            target_bpp=target_bpp, lambda_a=lambda_a,
        )
        d_fake = discriminator([x_hat, y_hat], training=False)
        g_loss = loss_fn.gan_generator_loss(d_fake, gan_loss_type)
        total = total + beta * g_loss

    amort_vars, entropy_vars = model.get_variable_groups()
    amort_grads = tape.gradient(total, amort_vars)
    entropy_grads = tape.gradient(total, entropy_vars)

    # Guard against NaN/Inf gradients from unstable early GAN steps
    amort_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in amort_grads
    ]
    entropy_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in entropy_grads
    ]

    amort_grads, _ = tf.clip_by_global_norm(amort_grads, 5.0)
    entropy_grads, _ = tf.clip_by_global_norm(entropy_grads, 5.0)

    amort_opt.apply_gradients(zip(amort_grads, amort_vars))
    entropy_opt.apply_gradients(zip(entropy_grads, entropy_vars))

    total_bpp = hyper_bpp + latent_bpp
    return total, g_loss, total_bpp


@tf.function
def discriminator_train_step(x, x_hat, y_hat, discriminator, disc_opt, gan_loss_type):
    """Train discriminator on real and generated images, conditioned on latents."""
    with tf.GradientTape() as tape:
        d_real = discriminator([x,                         y_hat], training=True)
        d_fake = discriminator([tf.stop_gradient(x_hat),   y_hat], training=True)
        d_loss = loss_fn.gan_discriminator_loss(d_real, d_fake, gan_loss_type)

    disc_grads = tape.gradient(d_loss, discriminator.trainable_variables)

    # Guard discriminator gradients — a freshly-init'd disc can spike on first steps
    disc_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in disc_grads
    ]
    disc_grads, _ = tf.clip_by_global_norm(disc_grads, 5.0)

    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return d_loss


# -------------------------------------------------------------------
# Main trainer
# -------------------------------------------------------------------

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    regime = REGIME_CONFIG[args.regime]
    target_bpp = regime["target_bpp"]
    lambda_a = regime["lambda_a"]

    # ----- Build model -----
    model = CompressionModel()

    # ----- Load warm-start checkpoint -----
    if args.warmstart and args.checkpoint:
        ckpt_load = tf.train.Checkpoint(model=model)
        ckpt_load.restore(args.checkpoint).expect_partial()
        print(f"Warm-started from {args.checkpoint}")

    # ----- Optimizers -----
    # Two-LR scheme for backbone: backbone LR is 1/10th of base
    backbone_vars, proj_vars = model.get_encoder_subgroups()

    amort_opt = tf.keras.optimizers.Adam(args.lr, beta_1=0.9, beta_2=0.999)
    entropy_opt = tf.keras.optimizers.Adam(args.lr, beta_1=0.9, beta_2=0.999)

    # ----- Discriminator (Phase 2 only) -----
    discriminator = None
    disc_opt = None
    gan_loss_type = "non_saturating"
    if args.model_type == "compression_gan":
        discriminator = build_discriminator(
            image_shape=(256, 256, 3),
            latent_channels=model.latent_channels,
        )
        disc_opt = tf.keras.optimizers.Adam(args.lr, beta_1=0.9, beta_2=0.999)

    # ----- Checkpoint management -----
    ckpt_objs = dict(model=model, amort_opt=amort_opt, entropy_opt=entropy_opt)
    if discriminator is not None:
        ckpt_objs["discriminator"] = discriminator
        ckpt_objs["disc_opt"] = disc_opt

    checkpoint = tf.train.Checkpoint(**ckpt_objs)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_dir,
        max_to_keep=5,
    )

    # Resume if existing checkpoint found
    if manager.latest_checkpoint and not args.warmstart:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint}")

    # ----- TensorBoard -----
    writer = tf.summary.create_file_writer(log_dir)

    # ----- Dataset -----
    ds = get_dataset(args.dataset_path, training=True,
                     batch_size=args.batch_size).repeat()

    # ----- LR schedule -----
    # Reduce base LR by 10× after lr_decay_step
    def get_lr(step):
        if step < args.lr_decay_step:
            return args.lr
        return args.lr * 0.1

    # ----- Main loop -----
    step = int(checkpoint.save_counter) * args.save_interval
    train_generator = True   # alternating flag for GAN mode

    print(f"Starting training at step {step} / {args.n_steps}")
    t0 = time.time()

    for batch in ds:
        if step >= args.n_steps:
            break

        # Update LR
        current_lr = get_lr(step)
        amort_opt.learning_rate.assign(current_lr)
        entropy_opt.learning_rate.assign(current_lr)

        # ----- Phase 1: compression only -----
        if args.model_type == "compression":
            total, r, d, p, bpp, x_hat = compression_train_step(
                batch, model, amort_opt, entropy_opt, target_bpp, lambda_a
            )
            if step % args.log_interval == 0:
                total_np = total.numpy()
                if not np.isfinite(total_np):
                    print(f"[{step:>7d}] WARNING: non-finite loss={total_np} "
                          f"(skipped — bad batch filtered by gradient guard)")
                else:
                    with writer.as_default():
                        tf.summary.scalar("loss/total", total, step=step)
                        tf.summary.scalar("loss/rate", r, step=step)
                        tf.summary.scalar("loss/distortion", d, step=step)
                        tf.summary.scalar("loss/perceptual", p, step=step)
                        tf.summary.scalar("bpp/total", bpp, step=step)
                        tf.summary.scalar("lr", current_lr, step=step)
                    elapsed = time.time() - t0
                    print(f"[{step:>7d}] loss={total_np:.4f}  bpp={bpp:.4f}  "
                          f"d={d:.4f}  p={p:.4f}  lr={current_lr:.2e}  "
                          f"t={elapsed:.1f}s")

        # ----- Phase 2: GAN fine-tuning -----
        elif args.model_type == "compression_gan":
            if train_generator:
                total, g_loss, bpp = generator_train_step(
                    batch, model, discriminator,
                    amort_opt, entropy_opt,
                    target_bpp, lambda_a,
                    beta=args.beta,
                    gan_loss_type=gan_loss_type,
                )
                # Cache x_hat and y_hat for discriminator step
                with tf.GradientTape() as _:
                    x_hat, _, _, y_hat = model(batch, training=False, return_latents=True)
                train_generator = False
            else:
                d_loss = discriminator_train_step(
                    batch, x_hat, y_hat, discriminator, disc_opt, gan_loss_type
                )
                train_generator = True

                if (step // 2) % args.log_interval == 0:
                    with writer.as_default():
                        tf.summary.scalar("loss/total", total, step=step)
                        tf.summary.scalar("loss/gan_g", g_loss, step=step)
                        tf.summary.scalar("loss/disc", d_loss, step=step)
                        tf.summary.scalar("bpp/total", bpp, step=step)
                    print(f"[{step:>7d}] total={total:.4f}  G={g_loss:.4f}  "
                          f"D={d_loss:.4f}  bpp={bpp:.4f}")

        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            save_path = manager.save(checkpoint_number=step)
            print(f"  Saved checkpoint: {save_path}")

        step += 1

    # Final save
    final_path = os.path.join(args.checkpoint_dir, f"final-{step}")
    checkpoint.write(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

    # Export density weights for ANS inference
    density_path = os.path.join(args.checkpoint_dir, "density_weights.npz")
    np.savez(density_path, **model.export_factorized_prior_weights())
    print(f"Density weights saved: {density_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train TFLite compression model")
    p.add_argument("--dataset_path", default="data/openimages")
    p.add_argument("--regime", choices=["low", "med", "high"], default="low")
    p.add_argument("--model_type", choices=["compression", "compression_gan"],
                   default="compression")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decay_step", type=int, default=500_000)
    p.add_argument("--beta", type=float, default=0.15,
                   help="GAN generator loss weight (Phase 2)")
    p.add_argument("--checkpoint_dir", default="experiments/tflite_low/")
    p.add_argument("--checkpoint", default=None,
                   help="Path to existing checkpoint for warm-start")
    p.add_argument("--warmstart", action="store_true")
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=10_000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
