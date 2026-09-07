# HSRNet

# HSRNet: Human-reconstructed and Dual-semantic Reinforcement Network for Occluded Text-to-Image Person Re-identification

by Li Yuan, Zihang Wu, Zhenfeng Zhao, Cong Wu, Mingfu Xiong, Junyi Zhu, Xiaokang Yang, Bin Sheng*
<img width="1026" height="536" alt="image" src="https://github.com/A01312/HSRNet/blob/main/img/framework.jpg" />

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

  * **ICFG-PEDES**
    
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
sh run.sh
```

## Evaluation

```bash
python test.py --config_file "$sub/configs.yaml"
```

## Results

<img width="865" height="841" alt="image" src="https://github.com/A01312/HSRNet/blob/main/img/Results.png" />

## Citation

```bibtex
@misc{hsrnet,
  title={HSRNet: Human-reconstructed and Dual-semantic Reinforcement Network for Occluded Text-to-Image Person Re-identification},
  author={Li Yuan, Zihang Wu, Zhenfeng Zhao, Cong Wu, Mingfu Xiong, Junyi Zhu, Xiaokang Yang, Bin Sheng},
  year={2026},
  note={Under review}
}
```

