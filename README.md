# UAAFusion
Codes for ***Deep Unfolding Multi-modal Image Fusion Network via Attribution Analysis (TCSVT 2024)***

[Haowen Bai](), [Zixiang Zhao](https://zhaozixiang1228.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Baisong Jiang](), [Lilun Deng](), [Yukun Cui](), [Shuang Xu](https://shuangxu96.github.io/), [Chunxia Zhang]().

-[*[Paper]*](https://ieeexplore.ieee.org/abstract/document/10769519)  
-[*[ArXiv]*](https://arxiv.org/abs/2502.01467)  

## Update
- [2025/2] Release training set and training code.
- [2025/1] Release inference code.

## Citation

```
@ARTICLE{10769519,
  author={Bai, Haowen and Zhao, Zixiang and Zhang, Jiangshe and Jiang, Baisong and Deng, Lilun and Cui, Yukun and Xu, Shuang and Zhang, Chunxia},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Deep Unfolding Multi-modal Image Fusion Network via Attribution Analysis}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Image fusion;Semantic segmentation;Optimization;Feature extraction;Analytical models;Visualization;Loss measurement;Image coding;Attention mechanisms;Transforms;Multi-modal image fusion;Algorithm unfolding;Attribution analysis;Memory augmentation},
  doi={10.1109/TCSVT.2024.3507540}}
```

## Abstract

Multi-modal image fusion synthesizes information from multiple sources into a single image, facilitating downstream tasks such as semantic segmentation. Current approaches primarily focus on acquiring informative fusion images at the visual display stratum through intricate mappings. Although some approaches attempt to jointly optimize image fusion and downstream tasks, these efforts often lack direct guidance or interaction, serving only to assist with a predefined fusion loss. To address this, we propose an “Unfolding Attribution Analysis Fusion network” (UAAFusion), using attribution analysis to tailor fused images more effectively for semantic segmentation, enhancing the interaction between the fusion and segmentation. Specifically, we utilize attribution analysis techniques to explore the contributions of semantic regions in the source images to task discrimination. At the same time, our fusion algorithm incorporates more beneficial features from the source images, thereby allowing the segmentation to guide the fusion process. Our method constructs a model-driven unfolding network that uses optimization objectives derived from attribution analysis, with an attribution fusion loss calculated from the current state of the segmentation network. We also develop a new pathway function for attribution analysis, specifically tailored to the fusion tasks in our unfolding network. An attribution attention mechanism is integrated at each network stage, allowing the fusion network to prioritize areas and pixels crucial for high-level recognition tasks. Additionally, to mitigate the information loss in traditional unfolding networks, a memory augmentation module is incorporated into our network to improve the information flow across various network layers. Extensive experiments demonstrate our method's superiority in image fusion and applicability to semantic segmentation.

## 🌐 Usage

### ⚙ Network Architecture

Our ReFusion is implemented in ``./nets/UAAFusion.py``.

### 🏊 Training
**1. Data Preparation**

Download the training set used in the article ([Google Drive](https://drive.google.com/file/d/1LHiN07gdgp-Ya6AI0EwFXalsT7DsG4IF/view?usp=drive_link)) and place it in ``./data/trainingdata``.

Or create a custom training set using ``./data/processing_img2patch.py `` after filling in lines 20-22.

**2. Training**

Ensure that the training data exists in ``./data/trainingdata``, and running 
```
python train.py
``` 

### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/UAAFusion.pth'``. 

``'./models/SS_pretrain.pth'`` is the pre-trained weight of the auxiliary segmentation model used for fusion training. 

**2. Test cases**

The 'test_cases' folder contains four cases. 
Running 
```
python test.py
``` 
will fuse these cases, and the fusion results will be saved in the ``./test_results`` folder.

**3. Test customization**

Modify the variables in 'test.py' as needed: 'path_ir' (the path of the infrared image), 'path_vi' (the path of the visible  image), and 'path_result' (the path to save the fusion result). Then run
```
python test.py
``` 
and the fusion results will be saved in the 'path_result'.