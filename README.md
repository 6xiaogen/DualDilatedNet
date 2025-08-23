Tooth Segmentation and Dental Crowding Diagnosis Using Two-Stage Dual-Dilated Graph Convolution

Abstract
Tooth segmentation and Diagnosis of dental crowding severity on 3D intraoral scan models are key processes for Computer-aided analysis of orthodontic models. Conventional methods are time-consuming, inefficient, and subjective, necessitating more efficient and intelligent approaches. Therefore, we propose a two-stage intelligent workflow.
In Stage 1, tooth segmentation is performed using an innovative dual-dilated graph convolutional network 1 (DDGCNet1). In Stage 2, Stage 1's output is converted to a point cloud, then processed by DDGCNet2 and post-processing to generate arch length discrepancy (ALD, an indicator of dental crowding). The encoding layers of the proposed networks embed a novel dual-dilated edgeconv module, effectively learning from local features and long-range contextual information of adjacent teeth. 

Requirements:

Python 3.10

PyTorch >= 2.1

CUDA >= 12.0

see requirements.txt for additional dependencies


<img width="779" height="369" alt="image" src="https://github.com/user-attachments/assets/990d7841-c355-4b13-bffe-4ffe6e81d79f" />




