# Agent-RewardBench


## 📄 Paper

📍 *Source code for Agent-RewardBench, ACL 2025 Main*

**Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents**



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

Please create a file named `config_private.py` and fill in the following information. Here is an example using **Qwen2-VL-7B-Instruct**.

```python
http_proxy = "your http_proxy"
https_proxy = "your https_proxy"

MODEL2URL = {"Qwen2-VL-7B-Instruct":"your api url"}
MODEL2KEY = {"Qwen2-VL-7B-Instruct": "your api key"}
MODEL2MODEL = {"Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct"}
```

### 3. Load Your Model

Using vLLM to load the model.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve Qwen2-VL-7B-Instruct --port 22347 --tensor-parallel-size 4 --trust-remote-code --limit_mm_per_prompt 'image=1'
```

### 4. Download Dataset

**Agent-RewardBench** is available on Hugging Face Datasets: [🤗 MultimodalAgent/Agent-RewardBench](https://huggingface.co/datasets/MultimodalAgent/Agent-RewardBench)

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("MultimodalAgent/Agent-RewardBench")
```

### 5. Run Your Model on Agent-RewardBench

```bash
python run.py
python performance.py
```

## 📦 Source Dataset

Agent-RewardBench involves Web (mobile, web, and desktop), Embodied (driving, house, Minecraft), and Travel, with data sources based on the following excellent works!

[Seeclick](https://arxiv.org/abs/2401.10935), [MFE-ETP](https://arxiv.org/abs/2407.05047), [Mind2Web](https://arxiv.org/abs/2306.06070), [PCA](https://arxiv.org/abs/2402.15527), [TravelPlanner](https://arxiv.org/abs/2402.01622?), [Pop-up attacks](https://arxiv.org/abs/2411.02391), [MSSBench](https://arxiv.org/abs/2410.06172)