# EvalCrafter: Benchmarking and Evaluating Large Video Generation Models 🎥📊

[Pages](http://evalcrafter.github.io) · [Paper@ArXiv](https://arxiv.org/abs/2310.11440) · [Prompt list](https://github.com/evalcrafter/EvalCrafter/blob/master/prompt700.txt) · [Huggingface Leaderboard](https://huggingface.co/spaces/AILab-CVC/EvalCrafter)

<div align="center">
<img src="https://github.com/evalcrafter/evalcrafter/assets/4397546/818c9b0d-35ac-4edf-aafc-ae17e92c6da5" width="250"/>
</div>

Welcome to EvalCrafter, a comprehensive evaluation toolkit for AI-generated videos. Our innovative framework assesses generative models across visual, content, and motion qualities using 17 objective metrics and subjective user opinions, providing a reliable ranking for state-of-the-art text-to-video generation models. Dive into the world of unified and user-centric evaluation with EvalCrafter! 🚀🌍📊

#### 🔥 2023/10/22: Release prompt list at [Prompt list](https://github.com/evalcrafter/EvalCrafter/blob/master/evalcrafter_prompt.txt)! You can generate the resulting video and send it to vinthony@gmail.com for evaluation!

#### 🔥 2024/01/10: Code and docker released!

#### 🔆 Join our Discord to enjoy free text-to-video generation and more: [![Discord](https://dcbadge.vercel.app/api/server/rrayYqZ4tf?style=flat)](https://discord.gg/rrayYqZ4tf)

#### 🔆 Watch our project for more details and findings.


## Installation 💻

Clone the repository:

   ```bash
   git clone https://github.com/evalcrafter/EvalCrafter
   ```

## Data Preparation 📚

Generate videos of your model using the 700 prompts provided in `prompt700.txt` or `./prompts` and organize them in the following structure:

```
/EvalCrafter/videos
├── 0000.mp4
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0699.mp4
```

## Pretrained Models 🧠
Please download  all the pretrained models following `./checkpoints/README.md` and organize them in this structure: 

```
/EvalCrafter/checkpoints/
├── bert-base-uncased
├── blip2-opt-2.7b
├── ckpt
├── clip-vit-base-patch32
├── clip_vit_base_patch32_config.json
├── clip_vit_base_patch32_pytorch_model.bin
├── Dover
├── FlowNet2_checkpoint.pth.tar
├── pt_inception-2015-12-05-6726825d.pth
├── RAFT
├── stable-diffusion-xl-base-1.0
├── tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth
├── vgg_face_weights.h5
└── VideoMAE
```

<!-- Alternatively, Download all the pretrained models from [Huggingface](https://huggingface.co/RaphaelLiu/EvalCrafter-Models) -->

## Setup 🛠️

### Run with Docker 🐳

1. Download the Docker image:

   ```
   docker pull bruceli1u1/evalcrafter:v1
   ```

2. Run the Docker container:

   ```
   docker run --runtime=nvidia -it --shm-size "15G" -v $your_local_path_to_EvalCrafter:$your_local_path_to_EvalCrafter \
       bruceliu1/evalcrafter:v1 bash
   ```
Please replace $your_local_path_to_EvalCrafter with your local path.

<!-- ### Run with Conda 🍃

1. Create the Conda environment and install dependencies:

   ```
   conda env create -f EvalCrafter_env.yml
   conda activate EvalCrafter
   cd $your_local_path_to_EvalCrafter$
   ``` -->

## Usage 💡

1. Run the complete evaluation pipeline:

   ```
   bash start.sh
   ```

2. To test a single metric, pick out the code for the metric in `start.sh`. For example, to test the Celebrity ID Score:

   ```
   cd /metrics/deepface
   python3 celebrity_id_score.py --dir_videos './videos'
   ```
   
  
## Acknowledgements 🙏

This work is based on the following open-source repositories:

- [deepface](https://github.com/serengil/deepface)
- [DOVER](https://github.com/teowu/DOVER-Dev)
- [mmaction2](https://github.com/open-mmlab/mmaction2)
- [CLIP](https://github.com/openai/CLIP)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [pytorch-gan-metrics](https://github.com/w86763777/pytorch-gan-metrics)
- [SDXL](https://github.com/Stability-AI/generative-models)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything)
- [BILIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- [HRS-Bench](https://github.com/eslambakr/HRS_benchmark)
- [fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency)

We would like to express our gratitude to the authors and contributors of these projects for making their code and models available. Your work has greatly contributed to the development of EvalCrafter. 🎉👏

---

Thank you for using EvalCrafter! We hope this toolkit helps you evaluate and improve your AI-generated videos with ease and efficiency. If you have any questions, suggestions, or feedback, please feel free to open an issue or submit a pull request. Happy evaluating! 🚀🌟

## Know More About Video Generation at:

- [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://github.com/AILab-CVC/VideoCrafter)
- [ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models](https://github.com/YingqingHe/ScaleCrafter)
- [TaleCrafter: Interactive Story Visualization with Multiple Characters](https://github.com/AILab-CVC/TaleCrafter)

