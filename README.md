# EvalCrafter: Benchmarking and Evaluating Large Video Generation Models 🎥📊

[Project Page](http://evalcrafter.github.io) · [Huggingface Leaderboard](https://huggingface.co/spaces/AILab-CVC/EvalCrafter)· [Paper@ArXiv](https://arxiv.org/abs/2310.11440) · [Prompt list](https://github.com/evalcrafter/EvalCrafter/blob/master/prompt700.txt) 


<div align="center">
<img src="https://github.com/evalcrafter/evalcrafter/assets/4397546/818c9b0d-35ac-4edf-aafc-ae17e92c6da5" width="250"/>
</div>

Welcome to EvalCrafter, a comprehensive evaluation toolkit for AI-generated videos. Our innovative framework assesses generative models across visual, content, and motion qualities using 17 objective metrics and subjective user opinions, providing a reliable ranking for state-of-the-art text-to-video generation models. Dive into the world of unified and user-centric evaluation with EvalCrafter! 🚀🌍📊

#### 🔥 2023/10/22: Release prompt list at [Prompt list](https://github.com/evalcrafter/EvalCrafter/blob/master/prompt700.txt)! You can generate the resulting video and send it to vinthony@gmail.com for evaluation!

#### 🔥 2024/01/10: Code and docker released!

#### 🔆 Join our Discord to enjoy free text-to-video generation and more: [![Discord](https://dcbadge.vercel.app/api/server/rrayYqZ4tf?style=flat)](https://discord.gg/rrayYqZ4tf)

#### 🔆 Watch our project for more details and findings.


## Installation 💻

Clone the repository:

   ```bash
   git clone https://github.com/evalcrafter/EvalCrafter
   cd EvalCrafter
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
Please download all checkpoints using 
```
cd checkpoints
bash download.sh
```

Alternatively, you can follow `./checkpoints/README.md` to download pretrained models for specific metrics.

Note: Please organize the pretrained models in this structure: 
```
/EvalCrafter/checkpoints/
├── bert-base-uncased
├── blip2-opt-2.7b
├── ckpt
├── clip-vit-base-patch32
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

### Download Docker Image  🐳

   ```
   docker pull bruceli1u1/evalcrafter:v1
   ```

## Usage 🚀

### Running the Whole Pipeline

1. Run with command line:

   ```
   docker run -it -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 \
      bash -c "source /opt/conda/bin/activate EvalCrafter \
         && bash $bash_file $EC_path $EC_path/videos"
   ```

   🔁 Please replace `$EC_path`, `$bash_file`, and `$dir_videos` with your local path to `EvalCrafter`, `EvalCrafter/start.sh`, and `EvalCrafter/videos`, respectively. 

Alternatively, you can:

2. Enter the Docker container and run:

   ```
   docker run -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 bash
   cd $EC_path
   bash start.sh $EC_path $dir_videos
   ```

### Running a Single Metric

🔧 To test a specific metric, pick out the code for the metric in `start.sh`. For example, to test the Celebrity ID Score:

   ```
   docker run -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 bash
   cd $EC_path
   cd /metrics/deepface
   python3 celebrity_id_score.py --dir_videos $dir_videos
   ```

<!-- ### Run with Conda 🍃

1. Create the Conda environment and install dependencies:

   ```
   conda env create -f EvalCrafter_env.yml
   conda activate EvalCrafter
   cd $EC_path$
   ``` -->



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

## Citation
If you find this repository helpful, please consider citing it in your research:

   ```
   @article{liu2023evalcrafter,
  title={Evalcrafter: Benchmarking and evaluating large video generation models},
  author={Liu, Yaofang and Cun, Xiaodong and Liu, Xuebo and Wang, Xintao and Zhang, Yong and Chen, Haoxin and Liu, Yang and Zeng, Tieyong and Chan, Raymond and Shan, Ying},
  journal={arXiv preprint arXiv:2310.11440},
  year={2023}
   }
   ```


## Know More About Video Generation at:

- [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://github.com/AILab-CVC/VideoCrafter)
- [ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models](https://github.com/YingqingHe/ScaleCrafter)
- [TaleCrafter: Interactive Story Visualization with Multiple Characters](https://github.com/AILab-CVC/TaleCrafter)

