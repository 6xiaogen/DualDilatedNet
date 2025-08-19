Tooth Segmentation and Dental Crowding Diagnosis Using Two-Stage Dual-Dilated Graph Convolutio

牙齿拥挤度测量是正畸治疗方案制定和拔牙必要性的关键诊断过程，传统测量方法耗时长、效率低以及具有主观不确定性等问题。本文提出了涉及两个阶段过程的智能化牙齿拥挤度测量方法，阶段1通过Dual-Dilated Graph Convolutional Network 1(DDGCNet1)实现牙齿分割，阶段2是在阶段1输出结果上，结合Dual-Dilated Graph Convolutional Network 2(DDGCNet2)和后处理拟合算法识别牙齿拥挤度。特别是，在提出的两种网络中嵌入了新颖的Dual-Dilated Edgeconv算法模块，通过丰富动态图结构的深层次信息，增强了局部和全局感受野，并采用注意力池化聚合图结构信息，使得网络充分学习牙齿局部区域特征（如牙窝沟等）和更远的邻牙信息。实验评估表明，所提改进网络的分割性能优于其它先进分割网络，且上下颌的拥挤度识别的平均绝对误差(MAE)分别为1.553mm和1.434mm，满足正畸诊疗的临床应用需求。

<img width="779" height="369" alt="image" src="https://github.com/user-attachments/assets/990d7841-c355-4b13-bffe-4ffe6e81d79f" />




