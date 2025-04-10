# Animatica-ml-engine
Animatica is my diploma project that generates videos from text and images using neural networks. It automates animation creation with image generation, image-to-video conversion, and post-processing

The Project is real time application in opencv using first order model

## Deploying on Local

---
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

Download cascade file, weights and model and save in folder named extract

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

    ```python image_animation.py -i path_to_input_file -c path_to_checkpoint```

    Example:

    ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar```

2. Run application from video file:

    ```python image_animation.py -i path_to_input_file -c path_to_checkpoint -v path_to_video_file```

   Example:

    ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar -v .\video_input\test1.mp4 ```

## Demo

---

![test demo](docs/demo.gif)

