<h2 align="center">CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning
</a>

<h5 align="center">

[Hongbo Jin](https://hongbo-jin.github.io/hongbo.github.io/)<sup>*</sup>,
[Ruyang Liu](https://scholar.google.com/citations?user=pZ3sWH0AAAAJ&hl=en&oi=ao)<sup>\*</sup>,
[Wenhao Zhang](https://openreview.net/profile?id=~Wenhao_Zhang10)<sup>\*</sup>, 
[Guibo Luo](https://www.ece.pku.edu.cn/info/1062/2227.htm)<sup></sup>,
[Ge Li](https://openreview.net/profile?id=~Ge_Li2)<sup>✉</sup>

<sup></sup>School of Electronic and Computer Science, Peking University<br>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2505.11830-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2505.11830)
</div>

<p align="center" width="100%">
<img src="./images/overall pipeline.png"  width="80%" height="80%">
</p>

## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the CoT-Vid folder:**
```bash
git clone https://github.com/Hongbo-Jin/CoT-Vid.git
cd CoT-Vid
```

#### 2. **Install the inference package:**
```bash
conda create -n cot-vid python=3.10 -y
conda activate cot-vid
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```


## SGLang for SpeedUp Inference and Deployment

We use [SGLang](https://github.com/sgl-project/sglang) to speed up inference and deployment of CoT-Vid. You could make CoT-Vid as a backend API service with SGLang.

**Prepare Environment**:
    Following the instruction in the [sglang](https://github.com/sgl-project/sglang?tab=readme-ov-file#install)



## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex
@misc{jin2025cotviddynamicchainofthoughtrouting,
      title={CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning}, 
      author={Hongbo Jin and Ruyang Liu and Wenhao Zhang and Guibo Luo and Ge Li},
      year={2025},
      eprint={2505.11830},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11830}, 
}

```

## Acknowledgement

- [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT): the codebase we built upon.
Great work!
- The `﻿lmms-eval` framework and its core contributors, including Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, and Kairui Hu, for their support on the evaluation side.

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)
