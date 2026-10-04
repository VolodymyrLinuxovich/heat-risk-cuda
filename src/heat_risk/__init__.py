"""heat_risk: heat hazard evaluation with three implementations.

* ``cpu``: NumPy reference.
* ``gpu_torch``: ordinary PyTorch library ops (a baseline, not a custom kernel).
* ``gpu_cuda``: a custom CUDA C++ kernel compiled at runtime with NVIDIA cuda.core (NVRTC).
"""

__version__ = "0.1.0"
