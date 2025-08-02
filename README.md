# Spatiotemporal GCN for Recommender Systems

This repository contains the implementation of a spatiotemporal graph-based recommender system using OPTICS cllustering and self-attention as described in our paper:

    Generalized Self-Attentive Spatiotemporal GCN with OPTICS Clustering for Recommendation Systems
    Saba Zolfaghari, Seyed Mohammad Hossein Hasheminejad
    Published in 2024 15th International Conference on Information and Knowledge Technology (IKT).
    DOI: 10.1109/IKT65497.2024.10892621

## Overview

This model predicts user-item ratings using:

    1-hop temporal subgraphs clustered via OPTICS

    Relational Graph Convolutional Network (RGCN) for spatial features

    LSTM for temporal modeling

    Self-attention to weight subgraph embeddings

    MLP for final rating prediction

The model evaluates performance with RMSE, MAE, Precision@K, Recall@K, NDCG@K, Coverage, and Hit Rate.

## Dataset

The implementation uses the MovieLens 100K dataset.
Files needed:

    u.data – user-item interactions with timestamps

## Requirements

Install dependencies with pip install. Example:

pip install surprise

⚠ Note: This code requires NumPy < 2.0 for compatibility with some dependencies.

pip install "numpy<2"

## Usage

Run the main script:

python -u 'Spatiotemporal GCN.py'

## Citation

If you use this code, please cite:

S. Zolfaghari and S. M. Hossein Hasheminejad, "Generalized Self-Attentive Spatiotemporal GCN with OPTICS Clustering for Recommendation Systems," 2024 15th International Conference on Information and Knowledge Technology (IKT), Isfahan, Iran, Islamic Republic of, 2024, pp. 85-90, doi: 10.1109/IKT65497.2024.10892621. keywords: {Knowledge engineering;Adaptation models;Accuracy;Attention mechanisms;Graph convolutional networks;Filtering;Predictive models;Optics;Spatiotemporal phenomena;Recommender systems;Recommender System;Spatiotemporal Graph Convolutional Network;OPTICS Clustering;Self Attention Mechanism},

