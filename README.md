# Animatica-ml-engine
Animatica is my diploma project that generates videos from images using neural networks. It automates animation creation with image generation, image-to-video conversion, and post-processing.

## Deploying on Local

### Step 1

Create a virtual environment.

###### *Linux/macOS:*

```bash
python3 -m venv venv
source venv/bin/activate
```

###### *Windows:*

```bash
python -m venv venv
source venv/Scripts/activate
```

### Step 2

```bash
cd app
pip install -r requirements.txt
```

### Step 3

Download pretrained models and save them in folder named extract

Checkpoints can be found under following link: 
[google-drive](https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH)
 or [yandex-disk](https://disk.yandex.ru/d/lEw8uRm140L_eQ).

###### *Linux/macOS:*

```bash
unzip checkpoints.zip
rm checkpoints.zip
```

###### *Windows:*

Unzip ***checkpoints.zip*** using unzipping software like *7zip*.

### Step 4

Run the project:

1. Run application from live camera:

    ```python .\animate_in_real_time.py -i path_to_input_file -c path_to_checkpoint```

    Example:

    ```python .\animate_in_real_time.py -i .\data\input\monalisa.png -c .\data\checkpoints\vox-cpk.pth.tar```

2. Run application from video file:

    ```python .\animate_in_real_time.py -i path_to_input_file -c path_to_checkpoint -v path_to_video_file```

   Example:

    ```python .\animate_in_real_time.py -i .\data\input\monalisa.png -c .\checkpoints\vox-cpk.pth.tar -v .\data\input\test.mp4 ```

## ~~Deploying via Docker~~

DOES NOT WORK!

### ~~Step 1~~

```bash
docker build -t animatica-ml-engine .
```

### ~~Step 2~~

~~Run container.~~

###### ~~With CUDA:~~
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 \
       	--env DISPLAY=$DISPLAY \
        --env="QT_X11_NO_MITSHM=1" \
        -v /dev/video0:/dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
        --gpus all -v "${PWD}:/app" \
       	-p 8888:8888 -p 6006:6006 \
        --name first-order-model \
	    first-order-model jupyter notebook --no-browser --port 8888 --ip=* --allow-root

```

###### ~~Without CUDA:~~
```bash
docker run --rm -it --device=/dev/video0:/dev/video0 \
       	--env DISPLAY=$DISPLAY \
        --env="QT_X11_NO_MITSHM=1" \
        -v /dev/video0:/dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
        -v "${PWD}:/app" \
       	-p 8888:8888 -p 6006:6006 \
        --name first-order-model \
	    first-order-model jupyter notebook --no-browser --port 8888 --ip=* --allow-root

```

## Demo

![test demo](docs/demo.gif)

## TODO
- [x] Add pre-commit.
- [ ] Add GitHub Actions.
- [ ] Add Docker.
- [ ] Optimize and refactor the code.
