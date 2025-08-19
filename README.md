<img width="779" height="369" alt="image" src="https://github.com/user-attachments/assets/aa039487-52d2-40bd-99b9-dc4a960f8339" /># Tooth Segmentation and Dental Crowding Diagnosis Using Two-Stage Dual-Dilated Graph Convolution
基于对偶-扩展图卷积网络的牙齿分割和牙齿拥挤识别

牙齿拥挤度测量是正畸治疗方案制定和拔牙必要性的关键诊断过程，传统测量方法耗时长、效率低以及具有主观不确定性等问题。本文提出了涉及两个阶段过程的智能化牙齿拥挤度测量方法，阶段1通过Dual-Dilated Graph Convolutional Network 1(DDGCNet1)实现牙齿分割，阶段2是在阶段1输出结果上，结合Dual-Dilated Graph Convolutional Network 2(DDGCNet2)和后处理拟合算法识别牙齿拥挤度。特别是，在提出的两种网络中嵌入了新颖的Dual-Dilated Edgeconv算法模块，通过丰富动态图结构的深层次信息，增强了局部和全局感受野，并采用注意力池化聚合图结构信息，使得网络充分学习牙齿局部区域特征（如牙窝沟等）和更远的邻牙信息。
