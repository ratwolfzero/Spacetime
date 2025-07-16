# 📐 Golomb Signal Spectral Analysis

This Python program generates **Golomb ruler signals**, performs **Fourier analysis**, and visualizes the **time-domain representation**, **FFT magnitude spectrum**, and **power spectrum**. It also explores the effects of **DC component removal** and **windowing** on spectral analysis.

---

## 📌 What Is a Golomb Ruler?

A **Golomb ruler** is a set of integers such that all pairwise differences are unique. These rulers are used in applications like:

* Radar/sonar
* Sensor array design
* Coding theory

The program implements a **greedy growing algorithm** to construct the first `n` marks of a Golomb ruler.

---

## 🔧 What the Program Does

1. **Generate a Golomb Ruler:**

   * Using a fast growing algorithm, a Golomb ruler `G` with `n` marks is created.

2. **Construct Binary Signal:**

   * A signal of length `G[-1] + 1` is created, placing 1s at Golomb ruler positions and 0s elsewhere.

3. **Optional DC Removal:**

   * The mean of the signal is optionally subtracted (i.e., DC removal) to focus on oscillatory content in the spectrum.

4. **Apply Spectral Window (Optional):**

   * Supported windows: `hanning`, `hamming`, `blackman`, `rectangular`.

5. **FFT and Power Spectrum Computation:**

   * Magnitude and power spectra are computed using NumPy’s FFT.

6. **Plotting:**

   * Three plots are shown:

     * Time-domain Golomb signal
     * FFT magnitude spectrum
     * Power spectrum (in log scale)

---

## 📈 What You See: Output Description

### 1. 🟦 **Golomb Signal (Time Domain)**

* A **stem plot** showing binary values (1s at ruler marks).
* If DC is removed, the mean is subtracted → some values become negative → shown as **red stems**.

**Interpretation:**

* Blue stems → 1s at Golomb positions
* Red stems (if any) → Negative amplitudes caused by mean subtraction

---

### 2. 🔵 **FFT Magnitude Spectrum**

* Shows how signal energy is distributed across normalized frequencies \[0, 0.5].
* Sharp spikes often indicate sparse, quasi-random frequency content.
* The **DC component (0 Hz)** represents the **average** of the signal.

---

### 3. 🔷 **Power Spectrum (Log Scale)**

* Power spectrum = square of magnitude.
* Plotted in logarithmic scale for better dynamic range visibility.
* Useful for observing dominant frequency bands.

---

## 🧠 Key Observations

### ✅ DC Component = `n`, Power = `n²`

If DC is **not removed**, the FFT’s **DC component** equals the number of Golomb marks:

$$
\text{DC Magnitude} = \sum x[k] = n
$$

$$
\text{DC Power} = n^2
$$

**Why?**
The input signal has exactly `n` ones and the rest zeros, so:

* The **mean value** of the signal is `n / L`, where `L = G[-1] + 1` is the signal length.
* The **FFT at index 0** sums all values, which equals `n`.
* The **power at DC** becomes `n²`.

**Printed Output Example:**

```
Magnitude of DC component (mag[0]): 10.0
Power of DC component (power[0]): 100.0
```

---

### 🔴 Red Stems in Time Plot (When DC is Removed)

When the DC component is removed:

* The signal's mean is subtracted.
* The `1`s become values below 1, and many entries become **negative**.
* These are plotted as **red stems** in the time-domain plot.

---

### 🔍 Why Remove DC?

Removing the DC component:

* Centers the signal around zero mean.
* Eliminates the dominant 0 Hz component in the spectrum.
* Allows better visibility of **oscillatory components** and **true spectral features**.

---

## 🪟 What Is Windowing?

Windowing shapes the signal before the FFT to reduce **spectral leakage** caused by sharp signal edges. Supported windows:

* **Hanning**
* **Hamming**
* **Blackman**
* **Rectangular** (i.e., no windowing)

---

## 🧪 Example Usage

Run the program with:

```bash
python golomb_fft.py
```

It will execute two examples:

1. **DC Removed, Hanning Window**
2. **DC Kept, Hanning Window**

Both display time-domain and spectral plots with annotations.

---

## ⚙️ Function Overview

### `golomb_grow(n)`

Generates the first `n` marks of a Golomb ruler using a growing difference-checking method.

### `create_signal_from_golomb(G, remove_dc)`

Creates a binary signal from ruler `G`. If `remove_dc=True`, the mean is subtracted and the signal is normalized.

### `compute_spectrum(signal, apply_window, window_type)`

Computes FFT, magnitude, and power spectra. Applies a window if requested.

### `plot_results(...)`

Displays:

* Time-domain stem plot (top)
* FFT magnitude (bottom-left)
* Power spectrum (bottom-right)

### `main(n, DC=False, windowing=False, window_type='hanning')`

Main entry point to run analysis with desired options:

* `n`: number of Golomb marks
* `DC=True`: keep DC component
* `windowing=True`: apply window before FFT
* `window_type`: choose window function

---

## 📚 Example Plot Layout

```
+--------------------------+
|     Golomb Signal        |
+-------------+------------+
|  FFT Mag    |   Power    |
|  Spectrum   |  Spectrum  |
+-------------+------------+
```

---

## 📎 Dependencies

* `numpy`
* `matplotlib`
* `numba`

Install with:

```bash
pip install numpy matplotlib numba
```

---

## 📤 Output Example (Console)

```bash
--- Example 3: DC Removed, Hanning Window Applied ---
Value of signal mean BEFORE FFT: 0.0
--- Applied hanning window ---
Magnitude of DC component (mag[0]): 2.17e-16
Power of DC component (power[0]): 4.72e-32

--- Example 4: DC Kept, Hanning Window Applied ---
Value of signal mean BEFORE FFT: 0.175
--- Applied hanning window ---
Magnitude of DC component (mag[0]): 10.0
Power of DC component (power[0]): 100.0
```

---

## 📌 Conclusion

This program demonstrates:

* Efficient generation of Golomb rulers
* Visual and spectral analysis of sparse signals
* Effects of DC removal and windowing on FFT
* Clear insights into frequency domain behavior

Ideal for learning, research, or signal processing demonstrations.

---

Would you like this saved as a file (`README.md`) or converted into a GitHub-ready repository structure?
