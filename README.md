# ENVI (ENVironment Imitation)
This repository provides experimental data and code of the paper below.
It includes:
- field operational test dataset of an autonomous robot vehicle,
- a virtual environment model generation algorithm using imitation learning (ENVI), and
- analysis for experiemt result data

## Publication
- Title: Virtual Environment Model Generation for CPS Goal Verification using Imitation Learning
- Authors: Yong-Jun Shin (ETRI, South Korea), Donghwan Shin (University of Sheffield, United Kingdom), Doo-Hwan Bae (KAIST, South Korea)
- Venue: ACM Transactions on Embedded Computing Systems
- Access: https://dl.acm.org/doi/full/10.1145/3633804

## Abstract
Cyber-Physical Systems (CPS) continuously interact with their physical environments through embedded software controllers that observe the environments and determine actions. Field Operational Tests (FOT) are essential to verify to what extent the CPS under analysis can achieve certain CPS goals, such as satisfying the safety and performance requirements, while interacting with the real operational environment. However, performing many FOTs to obtain statistically significant verification results is challenging due to its high cost and risk in practice. Simulation-based verification can be an alternative to address the challenge, but it still requires an accurate virtual environment model that can replace the real environment interacting with the CPS in a closed loop. 

In this paper, we propose ENVI (ENVironment Imitation), a novel approach to automatically generate an accurate virtual environment model, enabling efficient and accurate simulation-based CPS goal verification in practice.To do this, we first formally define the problem of the virtual environment model generation and solve it by leveraging Imitation Learning (IL), which has been actively studied in machine learning to learn complex behaviors from expert demonstrations. The key idea behind the model generation is to leverage IL for training a model that imitates the interactions between the CPS controller and its real environment as recorded in (possibly very small) FOT logs. We then statistically verify the goal achievement of the CPS by simulating it with the generated model. We empirically evaluate ENVI by applying it to the verification of two popular autonomous driving assistant systems. The results show that ENVI can reduce the cost of CPS goal verification while maintaining its accuracy by generating accurate environment models from only a few FOT logs. The use of IL in virtual environment model generation opens new research directions, further discussed at the end of the paper.

## Manual

### Key requirements
- python3.? or higher
- PyTorch

### Main
- main_acc.py: experiment code for the adaptive crusie control system (Case study 1)
- main_lk.py: experiment code for the lane keeping system (Case study 2)


## Contact
- Yong-Jun Shin (1st author): yjshin@etri.re.kr
- Donghwan Shin (Corresponding author): d.shin@sheffield.ac.uk
