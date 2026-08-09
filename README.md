# Riel Vision

Riel Vision is a lightweight computer-vision banknote counter for mixed Cambodian Riel (KHR) and US Dollar (USD) photos, designed for real-world retail environments.

> [!IMPORTANT]
> **To open the interactive presentation and live demo, double-click the batch file in the root folder:**
> **`Launch_Presentation.bat`**
> 
> *A local HTTP server is required to bypass browser CORS security policies when loading the 3D bill textures and ONNX model file.*

---

## Performance Benchmarks
These metrics measure how the hybrid one-detector Riel Vision model (trained on **2,416 real** and **3,096 synthetic** images) compares to previous baselines on our test slices:

| Test Set Slices | Khmer-US-currency (v7) | Real-only model | Riel Vision (Real + Synth) |
| :--- | :---: | :---: | :---: |
| **Easy / Clean** (separated bills) | 48.5% | 48.6% | **92.3%** |
| **Partially Blocked** (hand/finger occlusions) | 0.0% | 1.0% | **93.6%** |
| **Overlapping** (stacked/piled bills) | 1.6% | 6.0% | **56.6%** |

The public comparison is the [Khmer-US-currency model](https://universe.roboflow.com/robot-yfusg/khmer-us-currency-jofw1), published by the Roboflow Universe user **robot**. The **v7** label refers to the benchmark snapshot we used.

> [!WARNING]
> **Benchmark Bias Note**: These metrics are evaluated on static local test slices and reflect bias toward our specific validation conditions. Real-world performance under uncontrolled environments and arbitrary smartphone cameras will vary.

---

## Current Milestone & Honest Status

Riel Vision is currently a **research prototype**. We believe in being honest about what our model can and cannot do:

* **Our Approach**: We train a real-only model on real-world photos (which work well for clean, separated notes), and then use our custom WebGL synthesis engine to render the gaps—like overlapping banknotes and partially blocked banknotes—where real-world training data is scarce or difficult to label.
* **What Works Well**: Easy, clean, separated banknotes.
* **Active Challenges**: Dense, complex overlapping stacks, hand-held fan layouts, and extreme folds/crumples.

---

## Technology Stack

* **3D Synthesis Engine**: Powered by **Three.js** to generate exact-label training annotations (bounding boxes & segmentation masks) under controlled camera angles, lighting conditions, and note dirt/wear.
* **Edge Inference**: Running a lightweight **YOLO26n detector** directly in the browser via **ONNX Runtime Web** (WASM-based). The model is only **9.8 MB** on disk with **2.42M parameters**.

---

## Developer Directory Map
* **`Launch_Presentation.bat`**: Double-click to start python's local server and run the interactive dashboard.
* **`sample_dataset/`**: Folder containing visual examples of the different dataset categories used to train and evaluate the Riel Vision models (Real USD, Real KHR, Synthetic Overlapping, Synthetic Partially Blocked, and Real Fan Layout).
* **`model.md`**: The active working brain for plans, training runs, and promotion gates.
* **`AGENTS.md`**: Current project-scoped guidelines and constraints for developers.
