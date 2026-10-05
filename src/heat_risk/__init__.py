"""heat_risk: heat hazard evaluation with four implementations.

* ``cpu``: NumPy reference.
* ``cpu_cpp``: multithreaded C++17 on the host, compiled on first use and loaded with ctypes.
* ``gpu_torch``: ordinary PyTorch library ops (a baseline, not a custom kernel).
* ``gpu_cuda``: a custom CUDA C++ kernel compiled at runtime with NVIDIA cuda.core (NVRTC).
"""

__version__ = "0.1.0"
