# ViSA
Code for paper: View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification (CVPR2026)

## Requirements

Please refer to [INSTALL.md](./INSTALL.md).

Download the ViT-base Pre-trained model and modify the path. Line 13 in [configs](configs/CARGO/visa.yml):

> PRETRAIN_PATH: xxx

## Training & Testing

Training ViSA on the CARGO dataset with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/CARGO/visa.yml MODEL.DEVICE "cuda:0" SOLVER.IMS_PER_BATCH 256
```

Training ViSA on the CARGO dataset with 4 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/CARGO/visa.yml --num-gpus 4 SOLVER.IMS_PER_BATCH 512
```

Testing ViSA on the CARGO dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/CARGO/visa.yml --eval-only MODEL.WEIGHTS xxx 
```

## Citing ViSA
```text
@INPROCEEDINGS{zhang2026visa,
  author={Quan Zhang, Zeqiang Cai, Peiming Zhao, Jingze Wu, Cailun Wu, Hongbo Chen, Jianhuang Lai},
  booktitle={2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification}, 
  year={2026},
  volume={},
  number={},
  pages={1-10},
}

```
