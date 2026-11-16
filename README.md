### ARCT — Neural Network Architectures From Scratch

**ARCT** is a personal AI research and engineering project exploring neural network architectures built completely from scratch. Starting with zero-dependency, Python implementations and scaling up to vectorized NumPy engines and game-playing agents.


### Naming Choices & Architecture System

Every model in this repository follows a strict naming scheme.
**<Generation/Name><Language/Variant>-<Type> (<Inputs>, <Outputs>, <Parameters>)**

* **'ARCT' (Generation 1):** Pure Python implementations with **zero external libraries** (< 100 parameters).
* **'ARCT-R' (Rebound / Recoded):** Vectorized architectures powered by **NumPy** for deeper networks and up to 50,000+ parameters.
* **Language Suffixes:** py (Python), cpp (C++), j (Java).
* **Type Suffixes:** -gm (Gameplaying agents), -lm (Language models).


### Repository Structure

### ARCTpy/ (Pure Python — Zero Dependencies)
* **ARCT2_2_12.py — ARCTpy (2,2,12)**
  * **Architecture:** 2 -> 2 -> 2 Multi-Layer Perceptron (8 weights, 4 biases).
* **ARCT3_2_26.py — ARCTpy (3,2,26)**
  * **Architecture:** 3 -> 4 -> 2 Multi-Layer Perceptron (20 weights, 6 biases).

*Created by **Vexelpy** — MIT License*
