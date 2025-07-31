# BFSL client-selection algorithm
# Federated Learning Client Selection with Bandits under Wireless OFDMA

## Overview

This repository provides an implementation and simulation of client selection algorithms for **Federated Learning (FL)** operating over a **wireless OFDMA channel**, using a **Multi-Armed Bandit (MAB)** framework.

The central algorithm in this project reproduces and evaluates the method proposed in the following paper:

> **"Client Selection for Generalization in Accelerated Federated Learning: A Bandit Approach"**  
> *Dan Ben Ami, Kobi Cohen (Senior Member, IEEE), Qing Zhao (Fellow, IEEE)*  
> 1. School of Electrical and Computer Engineering, Ben-Gurion University of the Negev  
> 2. Department of Electrical and Computer Engineering, Cornell University  

The algorithm is compared against several existing MAB-based baselines for client selection.

---

## Motivation

Client selection in FL over wireless networks presents unique challenges:
- Limited bandwidth (OFDMA-based)
- Client and data heterogeneity
- Varying channel conditions
- Need for generalization and fast convergence

The selected clients influence not just model accuracy but also **training efficiency**, **communication cost**, and **generalization**. Using MAB frameworks enables adaptive and intelligent decision-making under uncertainty.

---

## Implemented Algorithms

| Algorithm         | Description |
|------------------|-------------|
| `BFSL` | Based on the paper above; selects clients to accelerate generalization using contextual MAB |
| `CS-UCB`         | Uses upper confidence bounds to balance exploration and exploitation |
| `CS-UCS-Q`       | Incorporates client quality and uncertainty into UCB |
| `RBCS-F`         | Rule-based selection with fairness constraints |
| `Random`         | Uniform random sampling of clients |
| `Power-of-Choice`| Selects the best among a random subset of candidates |

---



