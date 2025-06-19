# 🧠 Quantum Golomb Spacetime Simulator

![Quantum Golomb Spacetime](Quantum_Golomb_Spacetime_Analysis.png)

> *“Some say the universe began with a bang. This one began with `[0, 1]` and a poorly tuned random number generator.”*

## 📌 Overview

This project simulates a toy model of spacetime growth inspired by **Golomb rulers**, **quantum fluctuations**, **curved geometry**, and **causal networks**. While not physically rigorous, it offers a visually rich and conceptually intriguing framework to explore emergent structures from simple number-theoretic rules.

## 📐 What Is a Quantum Golomb Spacetime?

A **Golomb ruler** is a set of integers (marks) such that the distances between every pair are distinct. They are studied in combinatorics and optimization. In this simulator, they serve as the backbone of a fictional "spacetime," where:

* Every event (mark) is uniquely separated in time
* New marks are probabilistically added via a "quantum" growth rule
* Events experience **matter-curvature coupling**
* A **polar embedding** maps them into 2D space
* Emergent **density fields**, **curvature**, and **causal structure** are visualized

> ⚠️ Disclaimer: This is a *playful simulation*, not a model grounded in quantum field theory or general relativity.

---

## 🌱 Step 1: Birth of the Universe

The simulation begins with two seed marks:

```python
simulator = QuantumGolombSpacetime(initial_marks=[0, 1])
```

Marks are added iteratively using:

```python
simulator.quantum_growth(max_marks=40, temperature=0.1)
```

This growth process favors nearby valid candidates with a probability shaped by a **temperature parameter**:

* **Low T**: Conservative growth
* **High T**: Chaotic growth, high entropy

---

## 🔀 Step 2: Quantum Chaos via Matter-Curvature Coupling

Once the system contains enough marks, new additions are nudged by a synthetic interaction resembling gravity:

$$
\text{Potential} \sim \sum_i \frac{\rho_i}{(d_i^2 + \varepsilon)}
$$

Here, \$\rho\_i\$ is the "matter density" at an existing mark, and \$d\_i\$ is its distance from the candidate. This potential perturbs the position of new marks—sometimes yielding fractal behavior.

---

## 🌀 Step 3: Embedding in Polar Coordinates

Marks are embedded in polar space to visualize their geometric relationships:

$$
x = r \cdot \cos(\theta), \quad y = r \cdot \sin(\theta)
$$

Where:

* \$r\$ is derived from the mark value (optionally \$\log(1 + r)\$)
* \$\theta\$ is evenly distributed around the circle

This produces a **spacetime spiral** with angular and radial structure.

---

## 🌌 Step 4: Mass Density and Fractal Geometry

The polar coordinates are converted into a **2D density histogram** using Gaussian-smoothed binning. From this field:

* Local density “blobs” emerge
* The **fractal dimension** is estimated using a box-counting method:

$$
D = \lim_{\varepsilon \to 0} \frac{\log N(\varepsilon)}{\log(1/\varepsilon)}
$$

Where \$N(\varepsilon)\$ is the number of non-empty boxes at scale \$\varepsilon\$.

> Example output: `Estimated fractal dimension: 2.181`

---

## 🔊 Step 5: FFT Analysis

The spatial density is transformed into frequency space via a 2D FFT. This reveals:

* Radial and angular periodicities
* Self-similarity or emergent symmetries
* Clues to underlying order in the numeric chaos

---

## 🧭 Step 6: Causal Network Construction

We construct a **causal graph** where:

* Nodes = events
* Edges = directional, connecting future events within angular proximity
* Edge weight:

$$
w_{ij} = \frac{\rho_i}{\Delta t_{ij} \cdot (\Delta\theta_{ij} + \delta)}
$$

We then analyze:

* Connection density
* Average causal path length
* Correlation between **matter** and **causal influence**

---

## 📊 Physics Diagnostics (Toy Model Style)

### 🧬 Fluctuations

The simulator tracks average positional deviations from a linear growth path:

```python
Quantum Fluctuation = Mean(|∆position - 1|)
```

This captures the "quantum noise" introduced by randomness and curvature coupling.

---

### 🌐 Curvature-Matter Feedback

Using nearest-neighbor geometry in polar space, the **local curvature** is estimated and mapped to matter density:

* High curvature = higher local matter
* Feedback loop simulates gravity-like attraction

---

### ⚖️ Energy Conservation (Metaphorically)

We monitor a pseudo-conservation law:

$$
\text{Energy} \sim \frac{\text{Total Matter}}{\text{Average Curvature}}
$$

This serves as a qualitative check of system balance.

---

## 📈 Visualization Panels

The simulator produces a six-panel figure:

| Panel | Description                  |
| ----- | ---------------------------- |
| 📍    | **Polar Mark Embedding**     |
| 🌋    | **Mass Density Field**       |
| 🔊    | **FFT of Density Field**     |
| 🪐    | **Local Curvature Map**      |
| 🌐    | **Causal Network Layout**    |
| 🌈    | **Matter Density per Event** |

Each uses colorbars and consistent layouts for interpretability.

---

## 🧪 Summary of Findings

While the simulator is **not scientifically rigorous**, it highlights key ideas:

* Simple rules can produce complex, structured spacetime
* Golomb rulers encode uniqueness and hierarchy
* Matter and curvature can feedback via emergent mechanisms
* Causal structure reflects energy and temporal ordering

---

## 🧠 Final Thoughts

This project offers a sandbox for thinking creatively about space, time, and structure—without requiring a PhD in physics.

You may not find a TOE (Theory of Everything), but you might:

* Build intuition for emergent geometry
* Appreciate the beauty of discrete systems
* Discover humor in the cosmos

> *“My universe grew, curved, pulsed, and linked. All from `[0, 1]`. Just like ours—chaotic, kind of pretty, and mostly made up.”*

---

## 🛠️ Reqirements

```bash
pip install numpy matplotlib scipy scikit-learn networkx scikit-image
```

## ⚠️ Reminder

This is a creative simulation. Not suitable for replacing your cosmology textbook. But it might replace your existential dread with curiosity.
