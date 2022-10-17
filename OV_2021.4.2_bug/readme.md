##### System information (version)
- OpenVINO => 2021.4
- Operating System => Ubuntu - 20.04.4 LTS (Focal Fossa)
- Environment:
  1) conda==4.11.0
  2) python==3.8.13
  3) openvino-dev[onnx,pytorch,tensorflow2]==2021.4.2
  4) timm==0.6.7

##### Detailed description
ViT models converted from `timm` (pytorch-based) to `onnx` to `OpenVino` produce different outputs for the same input.

##### Steps to reproduce
use the script `reproduce_OV_error.py` to reproduce the error.