"""Baselines used by the paper's notebooks (not part of the accumap package).

The original accumap: similarity-weighted multidimensional scaling with a similarity-to-distance curve calibrated
on beads with known positions (Supplementary Note 3), and its idealized simulation.

  core         correct_locations (the weighted-MDS solver; uses the compiled accumap_rs extension when installed)
  curves       fit_curve, build_knn_mask, accumap_solve, ...
  simulation   simulate (idealized Gaussian-kernel simulation), cos_sim, per_bead_mse, jaccard_curve, ...

The solver is only practical with the Rust extension: `pip install maturin; cd baselines/accumap_rs;
maturin develop --release`. test_rust_vs_python.py checks it against the pure-Python fallback.
"""
