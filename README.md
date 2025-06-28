# AI_Waveform

This repository provides the source code for the paper:

**"Conditional Autoencoder for Generating Binary Neutron Star Waveforms with Tidal and Precession Effects"**

The project introduces a Conditional Autoencoder (cAE) for efficient and accurate generation of gravitational waveforms from Binary Neutron Star (BNS) mergers. It supports waveform reconstruction conditioned on physical parameters such as component masses, spins, and tidal deformabilities, and achieves high precision while maintaining strong generation efficiency.

## 🧠 Model Architecture

The figure below illustrates the architecture of the proposed Conditional Autoencoder (cAE) model for gravitational waveform generation. The model consists of three main components: (1) the autoencoder for waveform reconstruction, (2) the conditional encoder that maps physical parameters (e.g., masses, spins, tidal deformabilities) to latent space, and (3) the shared decoder used during inference to generate waveforms from encoded conditions. The encoder and decoder modules integrate convolutional layers, ResNet blocks, and Transformer blocks for temporal feature extraction and reconstruction.

![Model Architecture](./total_network.png)

*Figure: Architecture of the cAE model used for training and inference. The training stage includes both waveform autoencoding and latent supervision via physical parameters. At test time, only the condition encoder and decoder are used to generate waveforms from parameter input.*

---

## 🔧 Repository Structure

```text
AI_Waveform/
│
├── /data/                  # Waveform generation and preprocessing scripts
├── /models/                # Model architecture: encoder1, encoder2, decoder
├── /training/              # Training scripts and optimizer setup
├── /evaluation/            # Evaluation: mismatch, waveform overlap, timing tests
├── /examples/              # Sample usage: single and batch inference
├── cae_tensorflow_pseudocode.md
├── README.md
└── requirements.txt        # Environment dependencies
