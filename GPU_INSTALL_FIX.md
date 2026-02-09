# Corrected GPU Installation Instructions for TrustMark

## System Information
- CUDA Version: 12.4 (from nvidia-smi)
- GPUs Available:
  - GPU 0: RTX 3060 (12GB)
  - GPU 1: GTX 1060 3GB
- Conda Environment: trustmark
- Python: 3.10

## Problem with README Instructions
The README.md contains incorrect/outdated GPU setup instructions:
- Specifies `cudatoolkit=12.8` which doesn't exist
- Contradictory: installs PyTorch via conda then overwrites with pip

## Corrected Installation Steps

### Option 1: Install PyTorch with CUDA 12.1 (Recommended)

```bash
# Activate your conda environment
conda activate trustmark

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
pip install gradio
pip install omegaconf>=2.1
pip install lightning>=2.0
pip install six>=1.9
pip install einops>=0.4.0

# Install trustmark package
cd /d/ML/__PAID__Aava/steganography/trustmark/python
pip install -e .
```

### Option 2: Install PyTorch with CUDA 11.8 (Alternative)

If you have issues with CUDA 12.1, use CUDA 11.8 build:

```bash
conda activate trustmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Then install other dependencies as above
```

## Verification

After installation, verify CUDA is detected:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Expected output:
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 2
GPU 0: NVIDIA GeForce RTX 3060
GPU 1: NVIDIA GeForce GTX 1060 3GB
```

## Start Gradio App

```bash
cd /d/ML/__PAID__Aava/steganography/trustmark
python app/gradio_ui.py
```

The device selector should now show both GPUs with full specifications.

## Notes

- **CUDA 12.1 PyTorch is compatible with CUDA 12.4 drivers**
- The numpy version is pinned to 1.26.4 (latest 1.x) to ensure compatibility
- opencv-python 4.9.0.80 is compatible with numpy <2.0
- Your RTX 3060 (GPU 0) has 12GB memory and will perform better than the GTX 1060
