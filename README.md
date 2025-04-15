# Animatica-ml-engine
Animatica is my diploma project that generates image animation from videos in real time or not using neural networks (FOMM).
It automates animation creation with image generation, image-to-video conversion, and post-processing.

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

Unzip __checkpoints.zip__ using unzipping software like __7zip__.

### Step 4

__Use cases:__

1. Run the project from __Jupyter Notebook__ named __test.ipynb__.
2. Run the project using __CLI__ (Command Line Interface).
   
   __Examples:__

   ```bash
   python main.py --mode train --configs config.yaml
   ```
   ```bash
   python main.py --mode reconstruction --configs config.yaml --checkpoint path/to/ckpt
   ```
   ```bash
   python main.py --mode animate --configs config.yaml --checkpoint path/to/ckpt
   ```

## Demo

![test demo](docs/demo.gif)

## TODO
- [x] Add pre-commit.
- [ ] Add GitHub Actions.
- [ ] Optimize and refactor the code.
