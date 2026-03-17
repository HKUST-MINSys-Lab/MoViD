## Installation

### Pre-requirments  
1. Please make sure that you have properly installed the [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) before installation.

2. Please prepare the essential data for inference:   
To download SMPL body models (Neutral, Female, and Male), you need to register for [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify](https://smplify.is.tue.mpg.de/). The username and password for both homepages will be used while fetching the demo data.  
Next, run the following script to fetch demo data. This script will download all the required dependencies including trained models and demo videos.  
```bash
bash scripts/setup/fetch_demo_data.sh
```

### Usage
1. Prepare a CUDA 11.3 / Python 3.9 Docker image that includes the dependencies from this repository. If you already maintain a compatible local image, reuse it here.
```bash
docker pull <your-movid-image>
```

2. Run the code with docker environment:
```bash
cd /path/to/MoViD
docker run -v .:/code/ --rm <your-movid-image> python demo.py --video <input-video.mp4>
```
