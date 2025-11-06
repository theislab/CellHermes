

# <img src="https://github.com/bm2-lab/CellHermes/blob/main/img/Logo.png" width="50" height="50"> Language may be all omics needs: Harmonizing multimodal data for omics understanding with CellHermes
  <a href="https://huggingface.co/EthanGao123/CellHermes-v1.0"><img alt="Chat"
    src="https://img.shields.io/badge/🤖%20Model-CellHermes%20V1-536af5?color=536af5&logoColor=white"/></a>
  <a href="https://huggingface.co/collections/EthanGao123/cellhermes"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CellHermes_AI-ffc107?color=ffc107&logoColor=white" /></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-GNU-f5de53?&color=f5de53"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53"/></a>
  <br>
## 💡 Introduction 
This repository hosts the official implementation of CellHermes, a framework that can unify heterogeneous single-cell omics data by existing LLMs. Upon powerful capabilitz of LLMs, such as text-based understanding and reasoning, we can used it as encoder, predictor and explainer. This design allows CellHermes to span the entire research loop from representation learning to prediction and interpretability.
<p align="center"><img src="https://github.com/bm2-lab/CellHermes/blob/main/img/overview.png" alt="CellHermes" width="900px" /></p> 

## 🔧 Environment
Our experiments were conducted on python=3.10.15 and our CUDA version is 12.6.
We recommend using Anaconda / Miniconda to create a conda environment for using CellHermes. You can create a python environment using the following command:
```python
conda create -n CellHermes python==3.10.15
```

Then, you can activate the environment using and install the required packages:
```python
conda activate CellHermes
pip install -r requirement.txt
```
The training of CellHermes was performed by LLaMA-Factory (version: 0.9.1), so it is needed to configure the environment according to the environment of LLaMA-Factory. This can be done by switching to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) or directly use the following command:
```
cd LLama_factory_v0.9.1.dev0 
pip install -e ".[torch,metrics]"
```

### 🤖 Pretrained Checkpoint
We release these variants of ​​CellHermes​​. Please download to the `model_ckpt` directory.
| Model Name | Stage | Description |
|------------|-----------| -------|
| [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   | Base | The base LLM used in this study |
| [CellHermes](https://huggingface.co/)   | Pretraining | Spiking LLMs model on single-cell transcriptomic data and PPI network, simultaneously |
| [CellHermes-Multi-Task](https://huggingface.co/) | Instruction fine-tuning | Instruction-tuned model adapter with 7 databases across 10 tasks |
| [CellHermes-T-Cell-Reactivity](https://huggingface.co/) | Instruction fine-tuning | Instruction-tuned model adapter with T cell tumor-reactivity prediction task  |

### 📚 Data

We provide the dataset in [Zenodo](https://zenodo.org/). Please download the data to the project directory and use the following command to extract it:
```bash
cd data
unzip pretrain_datasets.zip
unzip multitask_datasets.zip
unzip perturbation_scaling_law_dataset.zip
unzip t_cell_reactivity_dataset.zip
unzip gene_level_downstream_tasks.zip
unzip cell_level_downstream_tasks.zip
unzip benchmarked_gene_embeddings.zip
unzip benchmarked_cell_embeddings.zip
```

### 🌟 Overview

The overall directory structure of the project is as follows:
```
├── 📂 scripts/                                 # source code
├── 📂 bash_config/                             # training & inference config 
├── 📂 data/                                    # datasets
│   ├── 📂 pretrain_datasets/                   # datasets for pretraining CellHermes
│   ├── 📂 multitask_datasets/                  # datasets for fine-tuning CellHermes for multi task prediction
│   ├── 📂 perturbation_scaling_law_dataset/    # datasets for fine-tuning CellHermes for testing scaling law on genetic perturbation prediction
│   ├── 📂 t_cell_reactivity_dataset/           # datasets for fine-tuning CellHermes for t cell tumor reactivity
│   ├── 📂 gene_level_downstream_tasks/         # datasets for gene level benchmarking datasets
│   ├── 📂 cell_level_downstream_tasks/         # datasets for cell level benchmarking datasets
│   ├── 📂 benchmarked_gene_embeddings/         # datasets of gene embeddings from various benchmarked models
│   └── 📂 benchmarked_cell_embeddings/         # datasets of cell embeddings from various benchmarked models on various datasets
├──  📂 model_ckpt/                             # store the pretrained checkpoints
│   ├── 📂 LLaMA-3.1-8B-Instruct/               # Base open-source LLM model
│   ├── 📂 CellHermes/                          # CellHermes model
│   ├── 📂 CellHermes-Multi-Task/               # Multi-task CellHermes model
└── └── 📂 CellHermes-T-Cell-Reactivity/        # T cell reactivity prediction model

```

### 🚀 Training

Model training is conducted on 2 NVIDIA RTX A6000 GPUs.

```bash
conda activate CellHermes
cd LLama_factory_v0.9.1.dev0 
bash ../bash_config/pretrain.sh
llamafactory-cli export ../bash_config/merge_lora_config.yaml
```
### 🔆 As an encoder
The following are commands for encoding biological entities by CellHermes, such as genes, cells and cell-specific genes.
#### Obtaining gene embeddding for a given gene
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "Gene BRCA1" \
                    -o ./output/gene_tmp_emb.pkl
```
#### Obtaining cell embeddding for a given cell transcriptomic information (gene rank in this case)
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "A cell with genes ranked by expression: MALAT1 TMSB4X B2M SRGN FTH1 BTG1 GNLY TPT1 EEF1A1 HLA-A ZFP36L2 PTMA HLA-B TMSB10 XCL1 PABPC1 ANXA1" \
                    -o ./output/cell_tmp_emb.pkl
```
#### Obtaining gene embedding for a given gene from a specific cell with its transcriptomic information (gene rank in this case)
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_encoder_for_embedding.py \
                    -m ./model_ckpt/CellHermes \
                    -i "A cell with genes ranked by expression: MALAT1 TMSB4X B2M RGS1 CCL3 CCL4 CD69 JUNB HSP90AA1 ZFP36 FTH1 DNAJB1 DUSP1 SAT1 CXCR4. In this cell, Gene BRCA1" \
                    -o ./output/cell_specific_gene_tmp_emb.pkl
```
### 🔆 As a predictor
The following are commands for fine-tuning CellHermes with multiple task datasets, such as perturbation prediction, cell fitness prediction, gene interaction prediction, etc. Users can change the `--dataset` parameter in `multitask_ft.sh` file to incorporate any dataset they want.
```bash
conda activate CellHermes
cd LLama_factory_v0.9.1.dev0 
bash ../bash_config/multitask_ft.sh
```
The following are commands for inference on the one of downstream testing datasets.
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_predictor_for_prediction.py \
                    -m ./model_ckpt/CellHermes \
                    -a ./model_ckpt/CellHermes-Multi-Task \
                    -i ./data/task_tmp.json \
                    -o ./output/task_tmp_predictions.jsonl
```

### 🔆 As an explainer
The following are commands for explaining the CellHermes's prediction results based on text-based reasoning.
```bash
conda activate CellHermes
python ./scripts/CellHermes_as_explainer_for_reasoning.py \
                    -m ./model_ckpt/CellHermes \
                    -a ./model_ckpt/CellHermes-T-Cell-Reactivity \
                    -i "Given a T cell from metastatic melanoma patients with its top 100 highly expressed gene list, ranked by expression level: RGS1 CCL3 CCL4 CD69 JUNB HSP90AA1. You think that this T cell is Reactive. Please explain your reasoning." \
                    -o ./output/cell_tmp_reasoning.pkl
```
### 🌻 Acknowledgement
We gratefully acknowledge the use some of codes from the following projects: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [scGPT](https://github.com/bowang-lab/scGPT), [GenePT](https://github.com/yiqunchen/GenePT). Our work builds upon their foundational contributions.

## 🔖 Citation  
## 😀 Contacts
 gao.yicheng.98@gmail.com
