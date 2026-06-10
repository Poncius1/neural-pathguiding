
import sys
import numpy as np
import torch
import mitsuba as mi

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Mitsuba:", mi.__version__)

try:
    mi.set_variant("cuda_ad_rgb")
    print("Mitsuba variant:", mi.variant())
except Exception as e:
    print("Could not set Mitsuba CUDA variant:")
    print(e)