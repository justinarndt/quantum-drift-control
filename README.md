# Quantum Drift Control (QDC)

Reference implementation for continuous-time quantum error correction using Neural Ordinary Differential Equations (Neural ODEs) and dynamic stabilizer reweighting.

## Overview

This repository contains the `drift_orchestrator` and `stim_integration` modules described in U.S. Patent Application No. **63/940,641** (Filed Dec 14, 2025). 

It addresses the spectral diffusion bottlenecks in superconducting processors (e.g., Sycamore, Willow) by modeling $T_1$ fluctuation as a stochastic Ornstein-Uhlenbeck process and integrating these priors into a Minimum Weight Perfect Matching (MWPM) decoder in real-time.

## Build Instructions

This project uses **Bazel** for hermetic builds and testing.

### Prerequisites
- Bazel 7.4.1+
- Python 3.10+
- Docker (Recommended)

### Running Tests
To verify the drift physics and solver integration:

```bash
bazel test //tests:drift_test
```

### Simulation
To execute a chunked memory lifetime experiment with streaming noise injection:

```bash
bazel run //src/simulation:stream_generator -- --total_duration_ms=100
```

## Patent Notice
Intellectual Property Rights Enforced. The architecture defined herein, specifically the coupling of Neural SDEs to fast-path FPGA/MWPM decoding, is the subject of pending patent claims. This code is provided for academic review and validation purposes only. Integration into proprietary control stacks requires a commercial license.

Correspondence: Justin Arndt - justinarndtai@gmail.com