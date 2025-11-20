# SINTHETIC PATH MAKER
This small pipline is just the start of a bigger project. The main aim is to generate METADATA from real financial datas. This is a very primitive version and it just needs as a test.
Validation in on it's way, but Heuristically it seems as a very good Idea to train ML on a specific ASSET rather than the mostly common used "general data" wich sometimes makes our model not proper for the specific ASSET behaviour. In this very first Version of SINTHETIC PATH MAKER there is just price in the output column, but this can easily change by futute new FEATURE generation. Nevertheless this outputs can have real world application by generating some more columns based on the Price (SMA20,SMA50, realized volatility and more).


This repository implements the pipeline described in  

**“SINTETIC PATH MAKER: From Price-Driven Weights to Synthetic Market Paths” (2025)**.

The goal is to generate synthetic financial time series guided by the local dynamics of a reference price.  
Given a price series \(P_t\), the code builds a discrete sampling distribution on time indices that emphasizes
zones of strong upward or downward movement, and uses it to propagate returns into realistic synthetic paths
for price and additional features (e.g. volume).

---

## Overview

The pipeline is lightweight, fully non-parametric, and designed to be:

- **Directional**: sampling probabilities depend on the sign and magnitude of the smoothed derivative of \(\log P\);
- **Interpretable**: every step (weights, empirical CDF, inverse transform) has a clear probabilistic meaning;
- **Reproducible**: seeds and configuration parameters fully determine each synthetic run;
- **Extensible**: the same machinery can be applied to alternative signals and multiple assets.

Typical use cases include:

- scenario exploration and stress testing;
- generating longer paths from short historical series;
- probing strategy robustness under controlled bullish/bearish biases.

---

## Method in a Nutshell

Given a price series \(P_t\), \(t = 0, \dots, N-1\):

1. **Smooth derivative of log-price**

   - Define an equispaced index grid \(y_i = y_0 + i\,\Delta y\).
   - Compute a smoothed derivative \(d_i \approx \frac{df}{dy}(y_i)\) of \(f_i = \log P_i\) using:
     - a Savitzky–Golay (SG) filter with user-selected `window` and `polyorder`, or  
     - a moving-average + finite-difference fallback.

2. **Directional, nonnegative weights**

   - Split the derivative into positive/negative components  
     \(u_i = \max(d_i, 0)\), \(v_i = \max(-d_i, 0)\).
   - Build raw weights
     \[
     s_i = \alpha\,u_i^\gamma + \beta\,v_i^\gamma + \varepsilon,\quad \gamma \ge 1,\ \alpha,\beta>0.
     \]
   - Apply a light triangular smoothing with parameter \(\rho \in [0,1]\), obtaining \(\tilde{s}\).

3. **Empirical CDF on an equispaced grid**

   - On the same grid \(\{y_i\}\), treat each cell as having mass proportional to \(\tilde{s}_i\).
   - Construct a **piecewise-constant density** and its corresponding CDF \(F(x)\), which is linear within each cell.
   - Build a **numerically robust inverse** \(F^{-1}(u)\) that handles zero-mass plateaus by skipping to the next positive-mass index.

4. **Index sampling and synthetic path construction**

   - Draw indices \(i_1, \dots, i_{T-1}\) either:
     - directly from the discrete probabilities \(p_i = \tilde{s}_i / \sum_k \tilde{s}_k\), or
     - via inverse-CDF sampling using \(F^{-1}\) for improved control and stratification.
   - Compute price returns
     \[
     r^{(P)}_t = \frac{P_t - P_{t-1}}{\max(P_{t-1}, 10^{-12})},\quad t\ge1.
     \]
   - Propagate returns into synthetic paths using one of two policies:
     - **Price-only returns** (`use_price_only = True`): all features evolve using the sampled price return.
     - **Feature-specific returns** (`use_price_only = False`): each feature uses its own historical return at the sampled index.

This yields synthetic trajectories that inherit the directional structure of the original series while remaining controllable through a small set of parameters.

---

## Repository Structure

A natural organization (reflected by the paper) is:

- **Weighting module**  
  Derivative estimation (Savitzky–Golay or fallback), construction of directional weights \(s_i\), and triangular smoothing.
- **CDF / inverse module**  
  Empirical piecewise-constant CDF on an equispaced grid and a robust inverse-transform sampler.
- **Sampling utilities**  
  Helpers for drawing indices (and optionally multinomial counts) from the designed distribution.
- **Synthetic path builder**  
  Logic to propagate returns, handle `use_price_only` vs feature-specific evolution, and choose the starting index.
- **I/O and plotting**  
  Loading CSV price/feature series, exporting synthetic paths, and generating figures (CDF, original vs synthetic price, etc.).

You can mirror this in your own `src/` tree, for example:

```text
src/
  weighting.py
  cdf_inverse.py
  sampling.py
  synthetic_paths.py
  io_utils.py

In very generalistic terms it takes
In order to use some csv you need to launch this command in bash
python3 src/main.py --csv ../SPX_randblock_1.csv --price-col Close --date-col Date
# in future i'm programming a new version with other variables
