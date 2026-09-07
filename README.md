# HSRNet

# HSRNet: Human-reconstructed and Dual-semantic Reinforcement Network for Occluded Text-to-Image Person Re-identification

by Li Yuan, Zihang Wu, Zhenfeng Zhao, Cong Wu, Mingfu Xiong, Junyi Zhu, Xiaokang Yang, Bin Sheng*


## Introduction
HSRNet is designed for Occluded Text-to-Image Person Re-identification.  
The framework introduces two key modules:

- **Human Structure-Perception Reconstruction (HSPR):**  Segments the human body into semantic regions, and achieves compensatory reconstruction of occluded areas via edge-associated branching and self-correcting mechanisms.
- **Dual-Semantic-Guided Reinforcement (DSGR):** Fuses visual and textual modalities with an adaptive weighting strategy to mine fine-grained semantics and enhance feature understanding of unoccluded regions.

## Installation
```bash
git clone https://github.com/A01312/HSRNet.git
cd HSRNet
```

## Environment
- Python 3.8.10
- PyTorch
- Ubuntu 22.04

## Datasets

*  **CUHK-PEDES**

    Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) 
    
    Organize them in `./dataset/CUHK-PEDES/` folder as follows:
    ~~~
    |-- dataset/
    |   |-- CUHK-PEDES/
    |       |-- imgs
                |-- cam_a
                |-- cam_b
                |-- ...
    |       |-- reid_raw.json
    |-- others/
    ~~~

  *  **ICFG-PEDES**

    Download the ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN)   

    Organize them in `./dataset/ICFG-PEDES/` folder as follows:

    ~~~
    |-- dataset/
    |   |-- ICFG-PEDES/
    |       |-- imgs
                |-- test
                |-- train 
    |       |-- ICFG-PEDES.json
    |-- others/
    ~~~

*  **RSTPReid**

    Download the RSTPReid dataset from [here](https://github.com/njtechcvlab/rstpreid-dataset)   

    Organize them in `./dataset/RSTPReid/` folder as follows:

    ~~~
    |-- dataset/
    |   |-- RSTPReid/
    |       |-- imgs
    |       |-- data_captions.json
    |-- others/
    ~~~

* **Occlusion Instance Augmentation**
  
   Same as [MGCC](https://github.com/littlexinyi/MGCC)
  
## Training

Train a model by:

```bash
python train.py --dataset llcm --gpu 0
```

Arguments:

- `--dataset`: which dataset to use, including `llcm`, `sysu`, or `regdb`.
- `--gpu`: which GPU to use.

## Test

Test a model on LLCM, SYSU-MM01, or RegDB dataset by:

```bash
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 0 --dataset llcm
```

Arguments:

- `--dataset`: which dataset to use, including `llcm`, `sysu`, or `regdb`.
- `--mode`: `all` or `indoor`, where `indoor` is only used for the SYSU-MM01 dataset.
- `--tvsearch`: whether to perform thermal-to-visible search, only used for the RegDB dataset.
- `--resume`: the saved model path.
- `--gpu`: which GPU to use.

## Results

<img width="865" height="841" alt="image" src="https://github.com/user-attachments/assets/dd6a496f-ac88-46d1-99e3-23b1c7aed291" />

<img width="808" height="434" alt="image" src="https://github.com/user-attachments/assets/d301cf44-6dd7-4e7d-aba4-3c399dba2e42" />

