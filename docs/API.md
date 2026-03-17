## Python API

To use the Python API, finish the basic installation first ([Installation](INSTALL.md) or [Docker](DOCKER.md)).

If you use Docker environment, please run:

```bash
cd /path/to/MoViD
docker run -it -v .:/code/ --rm <your-movid-image> python
```

The current compatibility API entrypoint is `movid_api.py`. You can call it like this:
```bash
from movid_api import MoViDAPI
movid_model = MoViDAPI()
input_video_path = '<input-video.mp4>'
results, tracking_results, slam_results = movid_model(input_video_path)
```
