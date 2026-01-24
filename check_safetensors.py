import safetensors.torch
import inspect

print(inspect.signature(safetensors.torch.load_model))
