
# CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning

<p align="center" width="100%">
<img src="./images/overall pipeline.png"  width="80%" height="80%">
</p>

## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
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

We use [SGLang](https://github.com/sgl-project/sglang) to speed up inference and deployment of LLaVA-NeXT. You could make LLaVA-NeXT as a backend API service with SGLang.

**Prepare Environment**:
    Following the instruction in the [sglang](https://github.com/sgl-project/sglang?tab=readme-ov-file#install)

### LLaVA-NeXT/OneVision

Checkout the HTTP Post/Get and SRT usage at [sglang/examples/runtime/llava_onevision](https://github.com/sgl-project/sglang/tree/main/examples/runtime/llava_onevision)

### LLaVA-NeXT (Video)

**Launch and Run on (K) Nodes**:
- Go to sglang project
    ```
    cd PATH_TO/sglang
    ```
- First node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K 0 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    (e.g. bash examples/usage/llava_video/srt_example_llava_v.sh K 0 examples/usage/llava_video/videos/Q98Z4OTh8RwmDonc.mp4 lmms-lab/LLaVA-NeXT-Video-7B-DPO 16)
    ```
- Second node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K 1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    ```
- The K node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K K-1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    ```


## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex


```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- The `﻿lmms-eval` framework and its core contributors, including Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, and Kairui Hu, for their support on the evaluation side.

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)
