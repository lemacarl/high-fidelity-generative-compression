"""
Are the TF prior and the ANS tables even using the same weights?

trace_hyper.py showed the TF factorized prior charging ~14.4 bits on channels
where the ANS coder charges 0.000 — a 100x disagreement between two
implementations of the same density. Removing the intermediate clip and fixing
the sigmoid cancellation did not move it, so the formulas are not the problem.

The remaining possibility is that they are reading different numbers:

  TF    CompressionModel.factorized_prior, restored from the checkpoint with
        .expect_partial(), which SILENCES unmatched-variable warnings. The
        H/a/b weights live in plain Python lists populated in build(); if
        checkpoint restore misses them they stay at random initialisation,
        and a random density assigns ~0 probability to z ~ 2.

  ANS   FactorizedPriorNumpy, read straight from density_weights.npz, which
        export_factorized_prior_weights() wrote at the end of training.

This compares the two weight sets directly, reports whether restore matched
every variable, and evaluates both CDF implementations on identical inputs for
one channel.

Usage:
    python -m tflite.trace_prior_weights \
        --checkpoint experiments/tflite_low_v2/final-2000000 \
        --density_weights experiments/tflite_low_v2/density_weights.npz \
        --channel 45
"""

import argparse
import numpy as np
import tensorflow as tf

from tflite.compression.entropy_models import FactorizedPriorNumpy
from tflite.model.compression_model import CompressionModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--density_weights", required=True)
    p.add_argument("--channel", type=int, default=45)
    args = p.parse_args()

    model = CompressionModel()
    _ = model(tf.zeros([1, 256, 256, 3]), training=False)

    # Snapshot the prior BEFORE restore, to tell "unchanged by restore" from
    # "restored to something that happens to look untrained".
    before = [w.numpy().copy() for w in model.factorized_prior.weights]

    status = tf.train.Checkpoint(model=model).restore(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")
    try:
        status.assert_existing_objects_matched()
        print("  assert_existing_objects_matched: PASS")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  assert_existing_objects_matched: FAIL\n    {exc}")

    after = [w.numpy().copy() for w in model.factorized_prior.weights]
    moved = [not np.array_equal(a, b) for a, b in zip(before, after)]
    print(f"\nPrior variables changed by restore: {sum(moved)}/{len(moved)}")
    for w, m in zip(model.factorized_prior.weights, moved):
        print(f"  {'changed' if m else 'UNCHANGED':<10} {w.name:<40} {w.shape}")
    if not any(moved):
        print("  => restore did not touch the prior; it is at random init")

    # What names does the checkpoint actually carry for the prior?
    print("\nCheckpoint entries matching 'factorized' / 'prior':")
    found = [(n, s) for n, s in tf.train.list_variables(args.checkpoint)
             if "factorized" in n.lower() or "prior" in n.lower()]
    for n, s in found[:20]:
        print(f"  {n}  {s}")
    if not found:
        print("  (none — the prior is absent from the checkpoint entirely)")

    # ---- Compare weights against density_weights.npz ----
    npz = dict(np.load(args.density_weights, allow_pickle=True))
    fp = FactorizedPriorNumpy.from_weights(npz)
    fp.build_tables()

    print(f"\nWeights: TF layer vs {args.density_weights}")
    n_filters = int(npz["n_filters"])
    for i in range(n_filters + 1):
        tf_H = tf.nn.softplus(model.factorized_prior._H[i]).numpy()
        tf_a = model.factorized_prior._a[i].numpy()
        tf_b = model.factorized_prior._b[i].numpy()
        for tag, t, n in (("H", tf_H, npz[f"H_{i}"]),
                          ("a", tf_a, npz[f"a_{i}"]),
                          ("b", tf_b, npz[f"b_{i}"])):
            if t.shape != n.shape:
                print(f"  {tag}_{i}: SHAPE {t.shape} vs {n.shape}")
                continue
            d = float(np.abs(t - n).max())
            flag = "MATCH" if d < 1e-5 else "DIFFER"
            print(f"  {tag}_{i}: max|Δ|={d:<12.6g} {flag}")

    # ---- Same inputs, both implementations, one channel ----
    c = args.channel
    xs = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    np_cdf = fp._cdf_channel(xs, c)

    C = model.factorized_prior.n_channels
    probe = np.zeros((1, 1, len(xs), C), dtype=np.float32)
    probe[0, 0, :, c] = xs
    tf_logits = model.factorized_prior._logits_cumulative(
        tf.constant(probe)
    ).numpy()
    tf_cdf = 1.0 / (1.0 + np.exp(-tf_logits[c, :, 0]))

    print(f"\nCDF for channel {c} at the same inputs")
    print(f"  {'x':>6}{'numpy (ANS)':>16}{'tf (estimate)':>16}")
    for j, xv in enumerate(xs):
        print(f"  {xv:>6.1f}{np_cdf[j]:>16.6g}{tf_cdf[j]:>16.6g}")
    print(f"\n  mass at z=1  numpy={np_cdf[1] - np_cdf[0]:.6g}"
          f"   tf={tf_cdf[1] - tf_cdf[0]:.6g}")


if __name__ == "__main__":
    main()
