# Agent-RewardBench


## 📄 Paper

**Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents**

**ACL 2025**


<p>
  <img src="docs/overview.png" width="95%" height="95%" />
</p>

**Agent-RewardBench** is a  benchmark designed to evaluate reward modeling in real-world multimodal agent scenarios. It covers three critical dimensions:

- **Perception**: web perception and embodied perception.
- **Planning**: web navigation, embodied intelligence and travel planning.
- **Safety**: web attacks and embodied safety.


<p>
  <img src="docs/bench.png" width="95%" height="95%" />
</p>


## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Quester-one/Agent-RewardBench.git
cd Agent-RewardBench
conda create -n agentrewardbench python=3.10
conda activate agentrewardbench
pip install -r requirements.txt
```


### 2. Config Your Model

```python
http_proxy = "your http_proxy"
https_proxy = "your https_proxy"

MODEL2URL = {"Qwen2-VL-7B-Instruct":"your api url"}
MODEL2KEY = {"Qwen2-VL-7B-Instruct": "your api key"}
MODEL2MODEL = {"Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct"}
```

### 3. Load Your Model

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve Qwen2-VL-7B-Instruct --port 22347 --tensor-parallel-size 4 --trust-remote-code --limit_mm_per_prompt 'image=1'
```

### 4. Download Dataset

**Agent-RewardBench** is available on Hugging Face Datasets: [Agent-RewardBench](https://huggingface.co/datasets/MultimodalAgent/Agent-RewardBench)

To load the dataset, use the following code:

```python
from datasets import load_dataset

dataset = load_dataset("MultimodalAgent/Agent-RewardBench")
```

### 5. Run



```python
python run.py
```