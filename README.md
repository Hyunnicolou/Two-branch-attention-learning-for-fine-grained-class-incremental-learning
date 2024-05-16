# Two-branch Attention Learning for Fine-grained Class Incremental Learning

This repository contains the implementation of the research paper titled "Two-branch Attention Learning for Fine-grained Class Incremental Learning".

## Authors

- Jiaqi Guo
- Guanqiu Qi
- Shuiqing Xie
- Xiangyuan Li

## Publication Details

- **Publication Date**: 2021/12/1
- **Journal**: Electronics
- **Volume**: 10
- **Issue**: 23
- **Pages**: 2987
- **Publisher**: MDPI
- **URL**: [https://www.mdpi.com/2079-9292/10/23/2987](https://www.mdpi.com/2079-9292/10/23/2987)

## Abstract

As a long-standing research area, class incremental learning (CIL) aims to effectively learn a unified classifier along with the growth of the number of classes. Due to the small inter-class variances and large intra-class variances, fine-grained visual categorization (FGVC) as a challenging visual task has not attracted enough attention in CIL. Therefore, the localization of critical regions specialized for fine-grained object recognition plays a crucial role in FGVC. Additionally, it is important to learn fine-grained features from critical regions in fine-grained CIL for the recognition of new object classes.

This paper designs a network architecture named two-branch attention learning network (TBAL-Net) for fine-grained CIL. TBAL-Net can localize critical regions and learn fine-grained feature representation by a lightweight attention module. An effective training framework is proposed for fine-grained CIL by integrating TBAL-Net into an effective CIL process. This framework is tested on three popular fine-grained object datasets, including CUB-200-2011, FGVC-Aircraft, and Stanford-Car. The comparative experimental results demonstrate that the proposed framework can achieve the state-of-the-art performance on the three fine-grained object datasets.

## Code Structure

- `utils_pytorch.py`: Contains utility functions for PyTorch.
- `class_incremental_imagenet.py`: Implements class incremental learning on the ImageNet dataset.
- `class_incremental_cosine_air.py`: Implements cosine similarity-based class incremental learning on the FGVC-Aircraft dataset.
- `class_incremental_cosine_car.py`: Implements cosine similarity-based class incremental learning on the Stanford-Car dataset.
- `eval_cumul_acc.py`: Evaluates cumulative accuracy.
- `class_incremental_cosine_gaborcnn_cub200.py`: Implements cosine similarity-based class incremental learning with Gabor CNN on the CUB-200-2011 dataset.
- `class_incremental_cosine_imagenet.py`: Implements cosine similarity-based class incremental learning on the ImageNet dataset.
- `class_incremental_cosine_tbal_cub200.py`: Implements the TBAL-Net for class incremental learning on the CUB-200-2011 dataset.
- `modified_resnet.py`: Contains a modified version of the ResNet model.
- `cbf_class_incremental_cosine_imagenet.py`: Implements class incremental learning with cosine similarity on the ImageNet dataset using CBF.
- `resnet.py`: Standard ResNet model implementation.
- `modified_linear.py`: Contains a modified linear model.
- `dataset_zzy_.py`: Handles dataset processing.
- `class_incremental_cub200.py`: Implements class incremental learning on the CUB-200-2011 dataset.
- `gen_resized_imagenet.py`: Generates resized ImageNet dataset.
- `class_incremental_cosine_cub200.py`: Implements cosine similarity-based class incremental learning on the CUB-200-2011 dataset.
- `gen_imagenet_subset.py`: Generates a subset of the ImageNet dataset.

## Usage

To run the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Hyunnicolou/Two-branch-attention-learning-for-fine-grained-class-incremental-learning.git
   cd Two-branch-attention-learning-for-fine-grained-class-incremental-learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the provided scripts for training and evaluation:
   ```bash
   bash run.sh
   ```

## Datasets

The framework is tested on three popular fine-grained object datasets:
- CUB-200-2011
- FGVC-Aircraft
- Stanford-Car

Please ensure you have downloaded the datasets and placed them in the appropriate directory before running the scripts.

## Citation

If you find this repository useful in your research, please cite our paper:

```
@article{Guo2021TBALNet,
  title={Two-branch attention learning for fine-grained class incremental learning},
  author={Jiaqi Guo, Guanqiu Qi, Shuiqing Xie, Xiangyuan Li},
  journal={Electronics},
  volume={10},
  issue={23},
  pages={2987},
  year={2021},
  publisher={MDPI},
  url={https://www.mdpi.com/2079-9292/10/23/2987}
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
