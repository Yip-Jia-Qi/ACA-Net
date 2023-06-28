# ACA-Net: Towards Lightweight Speaker Verification using Asymmetric Cross Attention
Pytorch Implementation of ACA-Net for Speaker Verification. This repository contains only the model, which can easily be adapted for most speaker verification training frameworks. Make sure to have both TDNN.py and ACANet.py in the same folder, and run the unit test contained at the end of ACANet.py to ensure everything is working.

# Abstract
In this paper, we propose ACA-Net, a lightweight, global context-aware speaker embedding extractor for Speaker Verification (SV) that improves upon existing work by using Asymmetric Cross Attention (ACA) to replace temporal pooling. ACA is able to distill large, variable-length sequences into small, fixed-sized latents by attending a small query to large key and value matrices. In ACA-Net, we build a Multi-Layer Aggregation (MLA) block using ACA to generate fixed-sized identity vectors from variable-length inputs. Through global attention, ACA-Net acts as an efficient global feature extractor that adapts to temporal variability unlike existing SV models that apply a fixed function for pooling over the temporal dimension which may obscure information about the signal's non-stationary temporal variability. Our experiments on the WSJ0-1talker show ACA-Net outperforms a strong baseline by 5% relative improvement in EER using only 1/5 of the parameters.

# Citing ACA-Net
Please, cite ACA-Net if you use it for your research or business.

```bibtex
@misc{yip2023acanet,
      title={ACA-Net: Towards Lightweight Speaker Verification using Asymmetric Cross Attention}, 
      author={Jia Qi Yip and Tuan Truong and Dianwen Ng and Chong Zhang and Yukun Ma and Trung Hieu Nguyen and Chongjia Ni and Shengkui Zhao and Eng Siong Chng and Bin Ma},
      year={2023},
      eprint={2305.12121},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
