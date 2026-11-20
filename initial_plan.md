# Initial Plan: Multi-Stream Digital Transmit Beamformer

ECE-GY 6463 Advanced Hardware Design, Fall 2026

## 1. Project Team

- _Member 1 full name_ (_NetID_)
- _Member 2 full name_ (_NetID_)

## 2. IP Definition

### What the IP does

The IP is the digital baseband of a **mmWave phased-array transmitter**. A
transmitter with `M` antennas and one DAC per antenna can steer its energy
digitally: each antenna gets its own complex-weighted copy of the data, so the
array radiates toward chosen directions. With **digital precoding**, the array
sends `L` independent data streams at the same time, each on its own beam (for
example, to `L` users in different directions). This is how 5G/6G base stations
do multi-user MIMO.

The IP takes `L` streams of complex data symbols (e.g., QAM) and produces the `M`
per-antenna sample streams that drive the DACs. Per block of symbols, it:

1. **precodes:** mixes the `L` streams into `M` antenna signals with a complex
   weight matrix `W` (`M × L`);
2. **pulse-shapes:** upsamples each antenna signal by 2 with a root-raised-cosine
   (RRC) FIR filter;
3. **calibrates:** multiplies each antenna signal by a complex gain that cancels
   the gain/phase mismatch between the RF transmit chains;
4. **saturates and monitors:** clips samples to the DAC range and reports, per
   antenna, the average power, peak power and number of clipped samples.

Design size: `M = 16` antennas (uniform linear array, half-wavelength spacing),
up to `L = 4` streams, blocks of up to `N = 1024` symbols, 33-tap RRC filter,
16-bit I/Q in and out.

### Interaction with the PS

- **Inputs (from PS):** symbol blocks in DDR, sent by AXI DMA; the precoder `W`,
  calibration gains `g` and filter taps `h`, written to registers; and the
  parameters `n_streams` and `block_len`.
- **Outputs (to PS):** the `M` antenna sample streams, written to DDR by AXI DMA
  (on an RFSoC they would go to the DACs instead), and per-antenna statistics.
- **Control:** start / done / idle handshake and a "clipping occurred" status
  flag, over AXI4-Lite.
- **Stays on the PS:** computing `W`. Conventional beams just need steering
  vectors; zero-forcing needs a small `L × L` matrix inverse once per channel
  update. Both are small, irregular and infrequent, so they run in Python on the
  ARM.

### Mathematical operations

All four operations are standard: a **complex matrix-vector multiply**, a
**polyphase interpolating FIR filter**, an **element-wise complex multiply**,
and **saturation plus power accumulation**. Python reference model (this is
also the golden model for the testbenches):

```python
import numpy as np

def transmit_beamformer(S, W, h, g, dac_max=1.0):
    """
    S : (L, N)  complex symbols, one row per stream
    W : (M, L)  complex precoder
    h : (T,)    real RRC filter taps, T = 33
    g : (M,)    complex per-antenna calibration gains
    Returns antenna samples (M, 2N) and per-antenna stats.
    """
    M, N = W.shape[0], S.shape[1]

    # 1. Precoding: complex matrix-vector product per symbol
    X = W @ S                                    # (M, N)

    # 2. Interpolate by 2 with RRC FIR: zero-stuff, then filter each antenna
    U = np.zeros((M, 2 * N), complex)
    U[:, ::2] = X
    Z = np.array([np.convolve(U[m], h)[:2 * N] for m in range(M)])

    # 3. Calibration: per-antenna complex gain
    Z = g[:, None] * Z

    # 4. Saturate to the DAC range, count clips, measure power
    clipped = (np.abs(Z.real) > dac_max) | (np.abs(Z.imag) > dac_max)
    Y = np.clip(Z.real, -dac_max, dac_max) + 1j * np.clip(Z.imag, -dac_max, dac_max)
    p = np.abs(Y) ** 2
    return Y, p.mean(axis=1), p.max(axis=1), clipped.sum(axis=1)
```

Polyphase form of step 2 (what the hardware computes, with no multiplies on the
inserted zeros). For each antenna `m` and input sample index `n`:

```
y[2n]   = sum_{k=0}^{16} h[2k]   * x[n-k]        # even phase, 17 taps
y[2n+1] = sum_{k=0}^{15} h[2k+1] * x[n-k]        # odd phase, 16 taps
```

How weights are made on the PS (for context, not in the IP):

```python
def steering(M, angles_deg):                     # half-wavelength ULA
    m = np.arange(M)[:, None]
    return np.exp(1j * np.pi * m * np.sin(np.deg2rad(angles_deg))[None, :])

W_conv = steering(16, [-40, -10, 20, 50]).conj() / 4     # conventional beams
W_zf   = H.conj().T @ np.linalg.inv(H @ H.conj().T)      # zero-forcing, H: (L, M)
```

Fixed-point formats (initial): symbols, weights, taps and outputs
`ap_fixed<16,1>`; precoder accumulators `ap_fixed<20,3>`; filter accumulators
`ap_fixed<24,4>`; power accumulators `ap_ufixed<48,12>`; output conversion uses
saturation (`AP_SAT`) and rounding.

### Why hardware suits this IP

- **Compute-intensive.** Per block of 1024 symbols: the precoder does
  `M·L·N` = 65k complex MACs, and the filter bank does
  `M · N · 33` ≈ 540k complex × real MACs, and calibration adds `M · 2N` = 33k
  complex multiplies: about 1.5M real multiplies in all.
  The output is `M × 2 = 32` complex samples per input symbol. A real array needs
  hundreds of MS/s per antenna, far beyond what the ARM can do in software.
- **Highly parallel.** The `M·L` precoder MACs are independent, and the 16
  antenna chains after the precoder are identical and independent. Both map onto
  replicated DSP pipelines running at II = 1.
- **Regular and streaming.** Every operation has a fixed schedule with no
  data-dependent branches, and data is used once, in order. Coefficients (`W`,
  `h`, `g`) are loaded once and held on chip, so the only memory traffic is the
  data stream.
- **Latency matters.** In a real transmitter, samples must reach the DACs at a
  fixed rate with no gaps, which needs a deterministic pipeline rather than
  software running under an OS.

## 3. IP Architecture

### Sub-modules

```
                    AXI4-Lite: params, W, h, g, start/done, status
                         │              │              │
 AXI DMA (MM2S)          ▼              ▼              ▼
 s_axis ─► [A] Input ─► [B] Precoder ─► [C] Polyphase ─► [D] Calibrate ─► [E] Output
          unpack        (M×L MACs)     RRC ×M (or ×M/F)  + saturate      pack
                                                         + monitor ×M     │
                                                                          ├─► x_axis ─► AXI DMA (S2MM)
                                                                          └─► stat_axis ─► AXI DMA (S2MM)
```

- **A. Input unpack.** Reads one 128-bit AXI4-Stream word per symbol time,
  splits it into `L = 4` complex 16-bit symbols, counts symbols up to `block_len`,
  and passes a block-end flag down the pipeline.
- **B. Precoder.** Computes `x_m = Σ_l W[m,l] · s_l` for all 16 antennas each
  symbol: 64 complex MACs with `W` in fully partitioned registers and an adder
  tree over `l`. `W` is double-buffered so the PS can load new weights during a
  block; they take effect at the next block. The symbol rate is half the output
  rate, so this module can run at II = 2 with half the multipliers.
- **C. Polyphase RRC filter bank.** One filter per antenna, each with a
  17-sample shift register. Each input sample produces two outputs (even and odd
  phase). Symmetric taps allow pre-adding sample pairs to halve the multipliers.
  This is the largest module, so it is written as a **single-antenna filter
  function replicated `M/F` times**, where the folding factor `F` (1, 2, 4)
  trades DSP slices for throughput. The AMD FIR Compiler IP is an alternative we
  may compare against.
- **D. Calibrate + saturate + monitor.** Per antenna: one complex multiply by
  `g_m`, conversion to 16-bit output with saturation, clip counting, and running
  sum and max of `|y|²`. Replicated `M` times, one small pipeline each.
- **E. Output pack.** Packs 16 complex samples into one 512-bit word for
  `x_axis` (or four 128-bit beats if the DMA is narrower). At block end, writes
  the per-antenna statistics to `stat_axis` and the clip flag to the register
  file, with `TLAST` on the last beat of each stream.

### Interfaces

| Connection | Type | Data carried | Width / rate |
|---|---|---|---|
| PS ↔ IP control | AXI4-Lite (`s_axilite`) | `n_streams`, `block_len`, start/done/idle, clip flag | 32-bit registers, once per block |
| PS → IP coefficients | AXI4-Lite (`s_axilite`) | `W` (64 complex), `g` (16 complex), `h` (17 unique taps) | ~100 words, once per configuration |
| DDR → A (`s_axis`) | AXI4-Stream via AXI DMA MM2S | `L` packed I/Q symbols, `TLAST` at block end | 128 bit, 1 beat per symbol |
| A → B | `hls::stream` FIFO | symbol vector `s[0..L-1]` + block-end flag | 1 per symbol |
| B → C | `hls::stream` FIFO, one per antenna | precoded sample `x_m` | 1 per symbol per antenna |
| C → D | `hls::stream` FIFO, one per antenna | filtered sample | 2 per symbol per antenna |
| D → E | `hls::stream` FIFO, one per antenna | 16-bit I/Q output sample; per-antenna stats at block end | 2 per symbol per antenna |
| E → DDR (`x_axis`) | AXI4-Stream via AXI DMA S2MM | 16 packed I/Q antenna samples, `TLAST` at block end | 512 bit, 2 beats per symbol |
| E → DDR (`stat_axis`) | AXI4-Stream via AXI DMA S2MM | avg power, peak power, clip count per antenna | 16 × 3 words per block |

No module uses shared memory (`m_axi`): every data path is read once, in order,
so streams are simpler and let all stages run at once. The top function uses
`#pragma HLS DATAFLOW`, so modules A–E run concurrently on successive symbols,
and the FIFOs absorb the 1:2 rate change at the filter.

### Why this modular structure

- **Incremental development.** Each module has its own C++ function and
  testbench against the matching step of the Python model (B vs. `W @ S`,
  C vs. `np.convolve` on the zero-stuffed signal, D vs. the clip / power code).
  We build and verify A→B first, then add C, then D and E, so there is always a
  working, tested pipeline.
- **Parallelism.** Splitting the work into stages lets DATAFLOW overlap them,
  and replicating the per-antenna modules (C, D) lets all antennas run side by
  side. One large function would serialize all of this.
- **Design-space exploration.** Because the filter bank is isolated and
  parameterized by the folding factor `F`, we can sweep `F`, bit widths, and
  polyphase vs. direct-form filtering, and measure DSP / BRAM use, II and
  throughput for each, without touching the rest of the design.
- **System-level checks.** End-to-end tests compute the radiated beam pattern
  and the signal each user receives from the IP output, and compare them with
  the floating-point model: main lobe at the target angle, sidelobe level, and,
  for zero-forcing, interference between users.
