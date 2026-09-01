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
from tflite.training.data_pipeline import get_dataset, CROP_SIZE
from tflite.training import losses as loss_fn

# -------------------------------------------------------------------
# PatchGAN discriminator (Phase 2 only — not exported to TFLite)
# -------------------------------------------------------------------

def build_discriminator(image_shape=(256, 256, 3), latent_channels=96,
                        sn_first=True, spectral_norm=True,
                        ctx_norm="layer"):
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
        sn_first:        Spectral-normalise the first tower conv. True matches
                         the reference; src/network/discriminator.py carries a
                         standing TODO to try False, which is the usual SN-GAN
                         practice of leaving the input layer free to set its
                         own gain. Try it if the tower still collapses.

    NOTE: the tower convs carry biases. They must. `nn.utils.spectral_norm`
    normalises the weight only and leaves the bias free, so the reference has
    960 unconstrained bias parameters. An earlier version of this port set
    use_bias=False, which — with the weight scale already pinned by spectral
    norm — left each layer no way to position its activations relative to the
    LeakyReLU kink. The tower degenerated toward a fixed linear map, emitted
    the same features for real and generated images, and the discriminator
    settled on constant output: d_loss = 2*ln2 = 1.3863, g_loss = ln2 = 0.6931,
    exactly, because 2*sigmoid(c) - 1 = 0 at c = 0 makes that a true
    stationary point rather than a slow region. It collapsed within ~1000
    discriminator updates in every run (v4-v7, high_gan), and since a constant
    generator loss has zero gradient, all of those runs were compression-only
    training wearing a GAN costume — confirmed by tflite_low_ft, which dropped
    the adversarial term entirely and reproduced v7 to three decimals.
    """
    image_in  = tf.keras.Input(shape=image_shape,              name="disc_image")
    latent_in = tf.keras.Input(shape=(16, 16, latent_channels), name="disc_latent")

    # ── Context path: latents → 12 channels → 256×256 ──────────────────────
    # Mirrors: self.context_conv + self.context_upsample in discriminator.py
    # The context path enters the concat with 12 channels against the
    # image's 3. Unnormalised latents from this port's encoder run |y| ~ 2.5
    # against |x| ~ 1, so the concatenated tensor is ~96% context energy —
    # and the context is identical in the real and fake branches, since both
    # are conditioned on the same y_hat. That leaves the discriminator hunting
    # a ~4% perturbation on a 96% shared signal, which is what it failed to do
    # across every probe: strong per-batch separation with a randomly-signed
    # ~0.005 offset, indifferent to spectral norm and to the loss function.
    # Normalising the latents brings the context in at unit scale.
    ctx_in = latent_in
    if ctx_norm == "layer":
        ctx_in = tf.keras.layers.LayerNormalization(
            name="disc_ctx_norm"
        )(ctx_in)
    ctx = tf.keras.layers.Conv2D(
        12, 3, padding="same", use_bias=True, name="disc_ctx_conv"
    )(ctx_in)
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
        # use_bias=True is load-bearing — see the NOTE in the docstring.
        conv = tf.keras.layers.Conv2D(
            f, 4, strides=2, padding="same",
            use_bias=True, name=f"disc_conv{i}"
        )
        if not spectral_norm or (i == 0 and not sn_first):
            x = conv(x)
        else:
            x = tf.keras.layers.SpectralNormalization(
                conv, name=f"disc_sn{i}"
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
        total_rd, r, d, p = loss_fn.total_compression_loss(
            x, x_hat, hyper_bpp, latent_bpp,
            target_bpp=target_bpp, lambda_a=lambda_a,
        )
        d_fake = discriminator([x_hat, y_hat], training=False)
        g_loss = loss_fn.gan_generator_loss(d_fake, gan_loss_type)
        total = total_rd + beta * g_loss

    amort_vars, entropy_vars = model.get_variable_groups()
    # The adversarial term shapes the encoder/decoder only. The entropy model
    # is driven by the rate-distortion loss alone, so GAN pressure cannot drag
    # the learned prior off its rate operating point.
    amort_grads = tape.gradient(total, amort_vars)
    entropy_grads = tape.gradient(total_rd, entropy_vars)

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
    # Return x_hat/y_hat so the discriminator step can reuse the *matching*
    # pair instead of recomputing it against a different batch.
    return total, g_loss, total_bpp, tf.stop_gradient(x_hat), tf.stop_gradient(y_hat)


@tf.function
def discriminator_train_step(x, x_hat, y_hat, discriminator, disc_opt, gan_loss_type):
    """Train discriminator on real and generated images, conditioned on latents.

    Real and generated go through in ONE concatenated forward pass, matching
    the reference (src/model.py discriminator_forward: `D_in = torch.cat(
    [x_real, x_gen], dim=0)`). Two separate calls are not equivalent here:
    keras SpectralNormalization runs a power iteration and assigns vector_u on
    every training=True call, so a second call re-normalises the kernel by a
    different sigma. d_real and d_fake would then be computed under different
    weights, and their difference — the only signal the discriminator has —
    would carry that weight change as noise.

    The latents are tiled, not repeat_interleaved. D_in is
    [real_0..real_{B-1}, gen_0..gen_{B-1}], so the matching latents are
    [y_0..y_{B-1}, y_0..y_{B-1}]. The reference uses repeat_interleave, which
    yields [y_0, y_0, y_1, y_1, ...] and pairs every image after the first
    with the wrong latents — the same defect that invalidated v4 here.
    """
    x_hat = tf.stop_gradient(x_hat)
    with tf.GradientTape() as tape:
        d_out = discriminator(
            [tf.concat([x, x_hat], axis=0), tf.tile(y_hat, [2, 1, 1, 1])],
            training=True,
        )
        d_real, d_fake = tf.split(d_out, 2, axis=0)
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
    # CLI overrides let a run be placed at a specific rate without editing
    # REGIME_CONFIG — needed to compare two models at MATCHED bitrate, since
    # quality metrics all improve with rate and are meaningless across
    # different ones.
    target_bpp = args.target_bpp if args.target_bpp is not None else regime["target_bpp"]
    lambda_a = args.lambda_a if args.lambda_a is not None else regime["lambda_a"]
    if args.target_bpp is not None or args.lambda_a is not None:
        print(f"Rate override: target_bpp={target_bpp}  lambda_a={lambda_a}  "
              f"(regime '{args.regime}' defaults: "
              f"{regime['target_bpp']}, {regime['lambda_a']})")

    # ----- Build model -----
    model = CompressionModel()

    # Force every sub-model to build its variables before anything restores or
    # seeds them. Sub-layers with a lazy build() — the factorized prior among
    # them — have no variables until the model is first called, so restoring
    # or assigning weights beforehand either defers silently or fails outright.
    _ = model(tf.zeros([1, CROP_SIZE, CROP_SIZE, 3]), training=False)

    # ----- Load warm-start checkpoint -----
    if args.warmstart and args.checkpoint:
        ckpt_load = tf.train.Checkpoint(model=model)
        ckpt_load.restore(args.checkpoint).expect_partial()
        print(f"Warm-started from {args.checkpoint}")

        # Checkpoints written before the factorized prior was tracked in the
        # object graph contain no prior, and expect_partial() hides that. The
        # warmstart would then silently begin phase 2 with a randomly
        # initialised entropy model, throwing away everything phase 1 learned
        # about the rate. Fall back to the density_weights.npz saved alongside
        # the checkpoint, which does hold the trained values.
        prior_in_ckpt = any(
            "factorized" in name.lower()
            for name, _ in tf.train.list_variables(args.checkpoint)
        )
        if prior_in_ckpt:
            print("  Factorized prior: restored from checkpoint")
        else:
            dw = args.prior_weights or os.path.join(
                os.path.dirname(args.checkpoint), "density_weights.npz"
            )
            if os.path.exists(dw):
                n_loaded = model.load_factorized_prior_weights(dw)
                print(f"  Factorized prior: absent from checkpoint — loaded "
                      f"{n_loaded} variables from {dw}")
            else:
                print("  WARNING: the warm-start checkpoint has no factorized "
                      "prior and no density_weights.npz was found beside it. "
                      "Phase 2 will start from a RANDOM entropy model and must "
                      "relearn the rate from scratch. Pass --prior_weights.")

    # ----- BatchNorm mode -----
    # Encoder BN is the only train/eval-dependent component; freezing it keeps
    # the rate estimate consistent between training and evaluation.
    if args.freeze_bn:
        n_frozen = model.freeze_batchnorm()
        print(f"Froze {n_frozen} encoder BatchNorm layers (inference mode)")

    # ----- Optimizers -----
    # Two-LR scheme for backbone: backbone LR is 1/10th of base
    backbone_vars, proj_vars = model.get_encoder_subgroups()

    # Adam beta_1=0.9 is fine for the pure rate-distortion phase but is a
    # well-known source of adversarial instability, so the GAN phase drops to
    # 0.5 and runs the generator at its own (lower) learning rate.
    is_gan = args.model_type == "compression_gan"
    adam_beta_1 = 0.5 if is_gan else 0.9
    if args.adam_beta_1 is not None:
        adam_beta_1 = args.adam_beta_1
    gen_lr = args.gen_lr if (is_gan and args.gen_lr is not None) else args.lr

    amort_opt = tf.keras.optimizers.Adam(gen_lr, beta_1=adam_beta_1, beta_2=0.999)
    entropy_opt = tf.keras.optimizers.Adam(gen_lr, beta_1=adam_beta_1, beta_2=0.999)

    # ----- Discriminator (Phase 2 only) -----
    discriminator = None
    disc_opt = None
    gan_loss_type = "non_saturating"
    if args.model_type == "compression_gan":
        discriminator = build_discriminator(
            image_shape=(256, 256, 3),
            latent_channels=model.latent_channels,
            sn_first=not args.disc_no_sn_first,
            spectral_norm=not args.disc_no_sn,
            ctx_norm=args.disc_ctx_norm,
        )
        # TTUR: the discriminator learns faster than the generator.
        disc_opt = tf.keras.optimizers.Adam(
            args.disc_lr, beta_1=adam_beta_1, beta_2=0.999
        )

    # ----- Materialise optimizer slots before any restore -----
    # Keras 3 optimizers create their momentum/velocity variables lazily, on
    # the first apply_gradients — which happens inside a @tf.function. A
    # checkpoint restore issued earlier is therefore still deferred at that
    # point, and firing it during graph construction raises NotFoundError on
    # keys like disc_opt/_variables/2. Building the optimizers here makes the
    # slots exist up front so the restore resolves eagerly and completely.
    amort_vars, entropy_vars = model.get_variable_groups()
    if not amort_opt.built:
        amort_opt.build(amort_vars)
    if not entropy_opt.built:
        entropy_opt.build(entropy_vars)
    if disc_opt is not None and not disc_opt.built:
        disc_opt.build(discriminator.trainable_variables)

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
        if args.reset_optimizers:
            # Restore weights only. Adam moments re-accumulate within a few
            # hundred steps, so this is a cheap escape hatch when optimizer
            # state in the checkpoint will not line up with a freshly built
            # optimizer.
            weights_only = dict(model=model)
            if discriminator is not None:
                weights_only["discriminator"] = discriminator
            tf.train.Checkpoint(**weights_only).restore(
                manager.latest_checkpoint
            ).expect_partial()
            print(f"Resumed WEIGHTS ONLY from {manager.latest_checkpoint} "
                  f"(optimizer state reset)")
        else:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"Resumed from {manager.latest_checkpoint}")

        # Same hazard as the warm-start path: if the factorized prior is not
        # in the checkpoint, restore leaves it at random initialisation and
        # training silently continues against a meaningless entropy model —
        # with every other weight and the optimizer state fully restored, so
        # nothing looks wrong. Seed it explicitly.
        prior_in_ckpt = any(
            "factorized" in name.lower()
            for name, _ in tf.train.list_variables(manager.latest_checkpoint)
        )
        if prior_in_ckpt:
            print("  Factorized prior: restored from checkpoint")
        elif args.prior_weights and os.path.exists(args.prior_weights):
            n_loaded = model.load_factorized_prior_weights(args.prior_weights)
            print(f"  Factorized prior: absent from checkpoint — loaded "
                  f"{n_loaded} variables from {args.prior_weights}")
            print("  NOTE: this is the prior as of the seed file, so any "
                  "adaptation it underwent before the interruption is lost. "
                  "It re-adapts over the remaining steps.")
        else:
            raise SystemExit(
                "Refusing to resume: the checkpoint contains no factorized "
                "prior and --prior_weights was not supplied. Resuming would "
                "train against a randomly-initialised entropy model.\n"
                "Pass --prior_weights <density_weights.npz>."
            )

    # ----- TensorBoard -----
    writer = tf.summary.create_file_writer(log_dir)

    # ----- Dataset -----
    ds = get_dataset(args.dataset_path, training=True,
                     batch_size=args.batch_size).repeat()

    # ----- LR schedule -----
    # Reduce base LR by 10× after lr_decay_step
    def get_lr(step):
        if step < args.lr_decay_step:
            return gen_lr
        return gen_lr * 0.1

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
                # x_hat/y_hat come straight out of the generator step, so they
                # are the noise-quantized tensors the generator was actually
                # trained on. `real_batch` is held alongside them because the
                # discriminator is conditional: its "real" branch must pair
                # these latents with the images they were encoded from.
                total, g_loss, bpp, x_hat, y_hat = generator_train_step(
                    batch, model, discriminator,
                    amort_opt, entropy_opt,
                    target_bpp, lambda_a,
                    beta=args.beta,
                    gan_loss_type=gan_loss_type,
                )
                real_batch = batch
                train_generator = False
            else:
                d_loss = discriminator_train_step(
                    real_batch, x_hat, y_hat, discriminator, disc_opt, gan_loss_type
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
    p.add_argument("--target_bpp", type=float, default=None,
                   help="Override the regime's target bitrate. The rate loss "
                        "switches from lambda_b to lambda_a above this value, "
                        "so the model settles near it — raise it to place a "
                        "run at a higher rate for a matched-bitrate comparison.")
    p.add_argument("--lambda_a", type=float, default=None,
                   help="Override the regime's rate penalty above target. "
                        "Lower means bits are cheaper, so the model spends more.")
    p.add_argument("--model_type", choices=["compression", "compression_gan"],
                   default="compression")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decay_step", type=int, default=500_000)
    p.add_argument("--beta", type=float, default=0.15,
                   help="GAN generator loss weight (Phase 2)")
    p.add_argument("--gen_lr", type=float, default=None,
                   help="Generator LR for Phase 2 (defaults to --lr). "
                        "Fine-tuning usually wants this well below --lr.")
    p.add_argument("--disc_lr", type=float, default=4e-4,
                   help="Discriminator LR for Phase 2 (TTUR; default 4e-4)")
    p.add_argument("--adam_beta_1", type=float, default=None,
                   help="Override Adam beta_1 for the generator/entropy "
                        "optimizers. Defaults to 0.9 for compression and 0.5 "
                        "for compression_gan. Set it explicitly when a "
                        "compression run has to match a GAN run's optimizer "
                        "so the two differ only in the adversarial term.")
    p.add_argument("--disc_ctx_norm", choices=["layer", "none"],
                   default="layer",
                   help="Normalise the latents before the discriminator's "
                        "context conv. The context enters the concat with 12 "
                        "channels to the image's 3, so unnormalised latents "
                        "make the input ~96%% context energy — identical in "
                        "both branches, drowning the signal the discriminator "
                        "needs. 'none' restores the pre-fix behaviour.")
    p.add_argument("--disc_no_sn", action="store_true",
                   help="Drop spectral norm from the whole tower. Diagnostic "
                        "rather than a setting to train with: it removes the "
                        "Lipschitz cap entirely, so if `sep` still will not "
                        "grow the obstacle is not the constraint and the "
                        "discriminator cannot see the difference at all.")
    p.add_argument("--disc_no_sn_first", action="store_true",
                   help="Drop spectral norm on the first discriminator conv, "
                        "letting the input layer set its own gain. Standard "
                        "SN-GAN practice and a standing TODO in "
                        "src/network/discriminator.py. Try it if the tower "
                        "still collapses with biases restored.")
    p.add_argument("--freeze_bn", action="store_true",
                   help="Run encoder BatchNorm in inference mode during "
                        "training so train and eval behave identically")
    p.add_argument("--checkpoint_dir", default="experiments/tflite_low/")
    p.add_argument("--checkpoint", default=None,
                   help="Path to existing checkpoint for warm-start")
    p.add_argument("--warmstart", action="store_true")
    p.add_argument("--reset_optimizers", action="store_true",
                   help="On resume, restore model and discriminator weights "
                        "but start the optimizers fresh. Use when optimizer "
                        "state in the checkpoint cannot be matched; Adam "
                        "moments rebuild in a few hundred steps.")
    p.add_argument("--prior_weights", default=None,
                   help="density_weights.npz to seed the factorized prior when "
                        "the warm-start checkpoint predates prior tracking. "
                        "Defaults to density_weights.npz beside --checkpoint.")
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=10_000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
