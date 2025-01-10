import onnx
from onnxconverter_common import float16

model = onnx.load(r"C:\Users\admin\Desktop\weights\minicoco\best32.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, r"C:\Users\admin\Desktop\weights\minicoco\best16.onnx")
