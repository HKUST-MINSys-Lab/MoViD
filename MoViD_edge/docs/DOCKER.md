## Installation

### Pre-requirments  
1. Please make sure that you have properly installed the [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) before installation.

2. Prepare the essential model files for inference and a local test video.
To download SMPL body models (Neutral, Female, and Male), you need to register for [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify](https://smplify.is.tue.mpg.de/).
If you also want action recognition on edge, see [docs/guides/action-recognition.md](guides/action-recognition.md).

### Usage
1. Prepare a CUDA 11.3 / Python 3.9 Docker image that includes the dependencies needed by `MoViD_edge`.
```bash
docker pull <your-movid-edge-image>
```

2. Run the code with docker environment:
```bash
cd /path/to/MoViD_edge
docker run -v .:/code/ --rm <your-movid-edge-image> python demo.py --video <input-video.mp4>
```
