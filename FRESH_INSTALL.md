# Fresh Installation Guide - Start Over

## Step 1: Delete Current Environment

```bash
# If the environment is currently active, deactivate it first
conda deactivate

# Delete the trustmark environment completely
conda env remove -n trustmark

# Verify it's deleted (should not appear in list)
conda env list
```

## Step 2: Create Fresh Environment with Python 3.10

```bash
# Create new conda environment with Python 3.10
conda create -n trustmark python=3.10 -y

# Activate the new environment
conda activate trustmark

# Verify you're in the right environment
which python
python --version  # Should show Python 3.10.x
```

## Step 3: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 (compatible with your CUDA 12.4 system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 4: Install FFmpeg (Required for Live Streaming Tab)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg

# Verify ffmpeg is available
ffmpeg -version
```

## Step 5: Install Dependencies

```bash
# Install numpy 1.26.4 (latest 1.x, required by trustmark)
pip install numpy==1.26.4

# Install OpenCV compatible with numpy <2.0
pip install opencv-python==4.9.0.80

# Install Gradio for the web UI
pip install gradio

# Install other required dependencies
pip install omegaconf>=2.1 lightning>=2.0 six>=1.9 einops>=0.4.0
```

## Step 6: Install TrustMark Package

```bash
# Navigate to the python directory
cd /d/ML/__PAID__Aava/steganography/trustmark/python

# Install in editable mode (so changes to code are reflected immediately)
pip install -e .
```

## Step 7: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Check TrustMark
python -c "from trustmark import TrustMark; print('TrustMark imported successfully')"

# Check Gradio
python -c "import gradio as gr; print(f'Gradio: {gr.__version__}')"

# Check OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check FFmpeg
ffmpeg -version
```

### Expected Verification Output:

```
PyTorch: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 2
  GPU 0: NVIDIA GeForce RTX 3060
  GPU 1: NVIDIA GeForce GTX 1060 3GB
TrustMark imported successfully
Gradio: 4.x.x
OpenCV: 4.9.0
```

## Step 8: Run the Gradio App

```bash
# Navigate to project root
cd /d/ML/__PAID__Aava/steganography/trustmark

# Start the Gradio server
python app/gradio_ui.py
```

The app will start on `http://0.0.0.0:7860`

You should see:
- Device selector showing both GPUs with full specs
- Model type selector with Q as default
- Ready to encode/decode videos

## Complete Script (Copy-Paste All at Once)

```bash
# Deactivate and delete old environment
conda deactivate
conda env remove -n trustmark -y

# Create fresh environment
conda create -n trustmark python=3.10 -y
conda activate trustmark

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install numpy==1.26.4 opencv-python==4.9.0.80 gradio av omegaconf lightning six einops

# Install TrustMark
cd /d/ML/__PAID__Aava/steganography/trustmark/python
pip install -e .

# Verify installation
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
python -c "from trustmark import TrustMark; print('TrustMark: OK')"
python -c "import gradio; print(f'Gradio: OK')"
python -c "import cv2; print(f'OpenCV: OK')"

echo "=== Installation Complete ==="
echo "To start the app:"
echo "  cd /d/ML/__PAID__Aava/steganography/trustmark"
echo "  python app/gradio_ui.py"
```

## Troubleshooting

### If conda env remove fails:
```bash
# Force remove
conda remove -n trustmark --all -y
```

### If you see "environment not found" when deactivating:
```bash
# Just skip the deactivate command and proceed with removal
conda env remove -n trustmark -y
```

### If pip install torch is slow:
- It's downloading ~2GB, this is normal
- Be patient, it may take 5-10 minutes depending on your connection

### If CUDA is not detected after installation:
```bash
# Check nvidia-smi still works
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
