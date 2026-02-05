# pytorch_training

Personal workspace of notebooks, helpers, and lab assignments from Coursera’s **“PyTorch for Deep Learning” Professional Certificate**.

> Note on content: this repo contains course/lab materials and generated artifacts. If you plan to publish it, review Coursera’s content/license terms and remove any files you’re not permitted to redistribute.

## Repository layout

- `pytorch_fundamentals/` — Course 1: fundamentals (tensors, basic NN building blocks, CNNs, data management, workflow)
- `Techniques_and_Ecosystem_tools/` — Course 2: TorchVision, Hugging Face (Transformers), hyperparameter optimization (Optuna), efficient training (Lightning), profiling
- `Advanced_Architectures_and_deployment/` — Course 3: custom architectures (Siamese/ResNet/DenseNet), vision interpretability, NLP seq2seq, deployment basics (MLflow, ONNX)

Most subfolders follow the naming pattern `C{course}_M{module}_Lab_{n}_*.ipynb` and include matching `helper_utils_*.py`.

## Quickstart (Windows)

### 1) Create a virtual environment

```bat
cd /d "<YOUR DIRECTORY>"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install PyTorch

PyTorch wheels differ by CPU vs CUDA. Use the official selector to get the right install command for your machine:
- https://pytorch.org/get-started/locally/

### 3) Install the rest of the Python dependencies

```bat
pip install -r requirements.txt
```

### 4) Run notebooks

Option A — VS Code:
- Install the VS Code **Python** and **Jupyter** extensions.
- Open any `.ipynb` and select the kernel from `.venv`.

Option B — Jupyter Lab:

```bat
python -m ipykernel install --user --name pytorch_training --display-name "pytorch_training (.venv)"
jupyter lab
```

## Notable labs (entry points)

- Fundamentals
  - `pytorch_fundamentals/getting_started_1/` — intro labs
  - `pytorch_fundamentals/pytorch_workflow_2/C1_M2_Lab_1_mnist_classifier.ipynb`
- Vision + TorchVision
  - `Techniques_and_Ecosystem_tools/working_with_images_using_torchvision/`
- NLP + Transformers (Hugging Face)
  - `Techniques_and_Ecosystem_tools/working_with_text_using_huggingface/`
- Hyperparameter optimization
  - `Techniques_and_Ecosystem_tools/Hyperparameter_optimization/C2_M1_Lab_3_Optuna.ipynb`
- Efficient training / profiling (Lightning)
  - `Techniques_and_Ecosystem_tools/Efficient_training_pipelines/`
  - Note: there is a folder named `lab_ssignment/` (typo in name) containing an assignment notebook.
- Model deployment
  - `Advanced_Architectures_and_deployment/Preparing_models_for_deployment/C3_M4_Lab_1_mlflow.ipynb`
  - `Advanced_Architectures_and_deployment/Preparing_models_for_deployment/C3_M4_Lab_2_onnx.ipynb`
- Generative models
  - `Advanced_Architectures_and_deployment/specialized_approaches_to_vision_in_pytorch/C3_M2_Lab_3_stable_diffusion.ipynb`

## Dependencies (high level)

This repo uses a mix of:
- Core: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`
- Training utilities: `torchmetrics`, `optuna`, `tqdm`
- NLP: `transformers` (+ model downloads)
- MLOps / deployment: `lightning`, `mlflow`, `onnx`, `onnxruntime`
- Vision utilities: `Pillow`, `ipywidgets`
- Optional: `diffusers` (Stable Diffusion)

See `requirements.txt` for a practical starting set.

## Tips / troubleshooting

- Spaces in paths are fine, but some tools behave better if you keep the repo in a shorter path.
- If widgets don’t render in Jupyter, ensure `ipywidgets` is installed and restart the kernel.
- Some notebooks download datasets/models on first run (TorchVision datasets, Hugging Face models). Expect network access and a bit of startup time.
