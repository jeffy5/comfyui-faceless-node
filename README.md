# Faceless Node for ComfyUI

> Next generation face toolkit for ComfyUI.

This project is under development. There may be compatibility issues in future upgrades. If you encounter any problems, please create an issue, thanks.

## Installation

### Install custom node

```bash
git clone https://github.com/jeffy5/comfyui-faceless-node
pip install -r requirements.txt
```

### Download model

All models is same as [facefusion](https://github.com/facefusion/facefusion) which can be found in [facefusion assets](https://github.com/facefusion/facefusion-assets).

Now it will use the following model by default. In the future, dynamic model selection will be supported. Please download the model based on the following directory structure.

Put all into directory `CmofyUI/models/faceless`.

```
|-- face_detector
|   `-- yoloface_8n.onnx
|-- face_landmarker
|   |-- 2dfan4.onnx
|   `-- face_landmarker_68_5.onnx
|-- face_recognizer
|   `-- arcface_w600k_r50.onnx
|-- face_swapper
|   |-- inswapper_128.onnx
|   `-- inswapper_128_fp16.onnx
|-- gender_age.onnx
`-- rmbg.pth
```

## Example workflows

### Image face swap

![Image Face Swap](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_swap_image.jpg)

### Video face swap

![Video Face Swap](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_swap_video.jpg)

### Load frames to preview image

![Load Frames To Preview](https://raw.githubusercontent.com/jeffy5/comfyui-faceless-node/main/.github/workflow_load_frames.jpg)

It doesn't support preview video for now. All video will be save in directory `output/faceless`.

## Thanks and Credits

Thanks to [Facefusion](https://github.com/facefusion/facefusion). This project is based on facefusion to implenment a special version for ComfyUI.
