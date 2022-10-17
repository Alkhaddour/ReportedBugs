import os
from pathlib import Path
 
import numpy as np
import torch
import timm
from openvino.inference_engine import IECore
 
 
# -------------- set paths --------------
TIMM_NAME='vit_base_patch16_224'
IMAGE_WIDTH = IMAGE_HEIGHT = 224
DIRECTORY_NAME = "debugOV"
BASE_MODEL_NAME = os.path.join(DIRECTORY_NAME, TIMM_NAME)
 
# Paths where PyTorch, ONNX and OpenVINO IR models will be stored
model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")
 
# -------------- Create Timm Model --------------
model = timm.create_model(model_name=TIMM_NAME,
                         pretrained=True,
                         num_classes=1)
 
# Save the model
model_path.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), str(model_path))
print(f"Model saved at {model_path}")
 
# -------------- ONNX Model Conversion --------------
if not onnx_path.exists():
   dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
 
   # for PyTorch>1.5.1
   with torch.no_grad():
       model=model.eval()
       torch.onnx.export(
           model,
           dummy_input,
           onnx_path,
           opset_version=12,
           do_constant_folding=True,
       )
       print(f"ONNX model exported to {onnx_path}.")
else:
   print(f"ONNX model {onnx_path} already exists.")
 
# -------------- Construct the command for Model Optimizer
mo_command = f"""mo
                --input_model "{onnx_path}"
                --input_shape "[1,3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]"
                --output_dir "{model_path.parent}"
                """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
 
 
if not ir_path.exists():
   print("Exporting ONNX model to IR... This may take a few minutes.")
   mo_result = os.system(mo_command)
   print(mo_result)
else:
   print(f"IR model {ir_path} already exists.")
 
# -------------- Try random image
image = np.ones((1, 3, IMAGE_WIDTH,IMAGE_HEIGHT))
print(image.shape)
 
# -------------- PyTorch inference
torch_outs = []
with torch.no_grad():
   for i in range(10):
       result_torch = model(torch.as_tensor(image).float())
       torch_outs.append(result_torch.item())
 
# -------------- OV inference
ie = IECore()
net_onnx = ie.read_network(model=onnx_path)
exec_net_onnx = ie.load_network(network=net_onnx, device_name="CPU")
 
input_layer_onnx = next(iter(exec_net_onnx.input_info))
output_layer_onnx = next(iter(exec_net_onnx.outputs))
 
# Run inference on the input image
outputs = []
for i in range(10):
    res_onnx = exec_net_onnx.infer(inputs={input_layer_onnx: image})
    res_onnx = res_onnx[output_layer_onnx]
    outputs.append(res_onnx[0][0])
 
print("PyTorch output:\n", torch_outs)
print("OV output:\n", outputs)
