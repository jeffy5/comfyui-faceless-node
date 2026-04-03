# Faceless Node for ComfyUI

> Next generation face toolkit for ComfyUI.

This project is under development. There may be compatibility issues in future upgrades. If you encounter any problems, please create an issue, thanks.

## Installation

### Install custom node

Install custom node in directory `ComfyUI/custom_nodes`.

```bash
git clone https://github.com/jeffy5/comfyui-faceless-node
pip install -r requirements.txt
```

### Setup conda

```bash
# Install conda
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Setup conda env
conda init --all
conda create --name facefusion python=3.10
conda activate facefusion

# Setup cuda
conda install conda-forge::cuda-runtime=12.4.1 cudnn=8.9.2.26 conda-forge::gputil=1.4.0
```

### Download model

All models is same as facefusion which can be found in [facefusion assets](https://github.com/facefusion/facefusion-assets).

Now it will use the following models by default. In the future, dynamic model selection will be supported. Please download the model based on the directory structure.

Put all into directory `ComfyUI/models/faceless`.

```
|-- face_detector
|   `-- yoloface_8n.onnx
|-- face_landmarker
|   |-- 2dfan4.onnx
|   `-- face_landmarker_68_5.onnx
|-- face_recognizer
|   `-- arcface_w600k_r50.onnx
├── face_restoration
│   └── gfpgan_1.4.onnx
|-- face_swapper
|   └── inswapper_128.onnx
|-- gender_age.onnx
`-- rmbg.pth
```

You can download models by running `download_models.py` script under directory `ComfyUI/custom_nodes/comfyui-faceless-node`.

```bash
# Install default required models
python download_models.py

# Install all models
python download_models.py --all
```

## Example workflows

You can find same example workflows in directory `examples`.

### Image face swap and face restore

![Image Face Swap And Face Restore](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_swap_image.jpg)

[image_face_swap_and_restore.json](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/examples/image_face_swap_and_restore.json)

### Video face swap and face restore

![Video Face Swap And Face Restore](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_swap_video.jpg)

[video_face_swap_and_restore.json](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/examples/video_face_swap_and_restore.json)

### Preview video frames

![Preview Video Frames](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_preview_video_frames.jpg)

[preview_video_frames.json](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/examples/preview_video_frames.json)

It doesn't support preview video for now. All video will be save in directory `output/faceless`.

## Thanks

Thanks to [Facefusion](https://github.com/facefusion/facefusion). This project is based on facefusion to implenment a special version for ComfyUI.
