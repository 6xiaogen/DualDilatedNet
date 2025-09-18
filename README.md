# <div align="center"><b>Tooth Segmentation and Dental Crowding Diagnosis Using Two-Stage Dual-Dilated Graph Convolution</b></div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/990d7841-c355-4b13-bffe-4ffe6e81d79f" width="850" height="420" />
</p>

## Abstract：
Tooth segmentation and diagnosis of dental crowding severity on 3D intraoral scan models are key processes for computer-aided analysis of orthodontic models. Conventional methods are time-consuming, inefficient, and subjective, necessitating more efficient and intelligent approaches. Therefore, we propose a two-stage intelligent workflow.  
In **Stage 1**, tooth segmentation is performed using an innovative dual-dilated graph convolutional network (**DDGCNet1**). In **Stage 2**, Stage 1's output is converted to a point cloud, then processed by **DDGCNet2** and post-processing to generate arch length discrepancy (ALD, an indicator of dental crowding). The encoding layers of the proposed networks embed a novel dual-dilated edgeconv module, effectively learning from local features and long-range contextual information of adjacent teeth.  

## Requirements：
```python
Python 3.10
PyTorch >= 2.1
CUDA >= 12.0
see requirements.txt for additional dependencies
```

## How to use
Our code is implemented based on PyTorch. **DDGCNet1** is used for tooth segmentation, while **DDGCNet2** is designed for crown segmentation based on both dental arch boundaries and crown width. Users can set up a PyTorch environment and install the dependencies listed in the requirements.txt file to train the models themselves. Alternatively, pre-trained models can be requested via email at han1024@nuaa.edu.cn for experimental use.

## Local Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Data
**Tooth Segmentation**:The data used in this project is the Teeth3DS dataset, which can be downloaded from [here](https://github.com/abenhamadou/3DTeethSeg22_challenge) <br>
The data should be placed in the data directory in the following structure:
```
data
    |3dteethseg
        | raw
            | lower
            | upper
            | private-testing-set.txt
            | public-training-set-1.txt
            | public-training-set-2.txt
            | testing_lower.txt
            | testing_upper.txt
            | training_lower.txt
            | training_upper.txt

```

**Arch-bounded crown segmentation**: The dataset is created by us, and we have made several sets available for download. If readers wish to obtain the dataset we created, they can contact us via email at han1024@nuaa.edu.cn.
The "data" file contains network data, with train_data and test_data representing the data used during the training and testing processes, respectively. The data for each tooth model is saved in a separate txt file, with the content as follows:
```python
29.343200 10.423900 -90.281000 0.004916 -0.002189 0.003073 3.000000 0
29.380800 10.475500 -90.313700 0.012912 -0.004292 0.007224 3.000000 0
29.389800 10.527000 -90.289500 0.010063 -0.005052 0.004136 3.000000 0
29.419200 10.580700 -90.297000 0.005329 -0.002501 0.002165 3.000000 0
…………
-18.442100 1.726450 -85.220300 0.012413 0.006148 0.010642 12.000000 1
-18.438900 1.998520 -86.188100 0.006604 0.013450 0.010322 12.000000 1
-18.434300 1.895030 -85.473100 0.002395 0.005126 0.000941 12.000000 1
…………
```
The second-to-last data point is the tooth number, and the last number represents either the internal or external label, with the dental arch serving as the boundary.

**Width-bounded crown segmentation**：As above.

## Code Explanation
The "dataset" folder in DDGCNet1 mainly handles the processing of dental mesh data, extracting data from OBJ files and downsampling to the required number of meshes. The "data_utils" folder in DDGCNet2 mainly deals with the processing of dental crown point clouds, including downsampling, etc. The "models" folder contains the source code for the network.

## Special Thanks
The code in this repository is based on the following repositories:
- [DGCNN](https://github.com/antao97/dgcnn.pytorch)
- [dilated_tooth_seg_net](https://github.com/LucasKre/dilated_tooth_seg_net.git)
- [PointNet2](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)







