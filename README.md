# climBright
An App for rock climbers.
- Upload images of climbing holds to classify them.
- Upload wall photos to detect holds and get climbing route suggestions.
- Built with FastAPI, MongoDB, YOLOv8, ConvNeXt, and many climbing enthusiasts.

## Run Everything (Local)

### 0) Prereqs
- MongoDB server (`mongod`)
    - macOS: `brew install mongodb-community@7.0`
    - Windows: `choco install mongodb`
- Python 3.10+ and Node 18+

### 1) Put model weights in place
- ConvNeXt classifier: place `best_convnext_two_phase.pt` in this folder (same level as `main.py`).
- YOLO detector weights: default path is `runs/detect/train2/weights/best.pt`.

If your files live somewhere else, set env vars when starting FastAPI:
- `CONVNEXT_MODEL_PATH=/absolute/path/to/best_convnext_two_phase.pt`
- `YOLO_MODEL_PATH=/absolute/path/to/best.pt`

### Note: Make a python venv (optional but recommended)

MacOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

Windows
```bash
python -m venv venv
venv\Scripts\activate.ps1 # Windows PowerShell
```

### 2) Install Python deps
```bash
pip3 install -r requirements.txt
```

### 3) Start MongoDB (Terminal 1)
```bash
mongod --dbpath ./db/mongo --bind_ip 127.0.0.1 --port 2701
```

### 4) Start FastAPI on port 9000 (Terminal 2)
```bash
uvicorn main:app --reload --port 9000
```

### 5) Start the web app (Terminal 3)
```bash
cd frontend
npm install
npm run start
```

# MAKE SURE TO ADD A .env FILE IN THE FRONTEND FOLDER

Open:
- `http://127.0.0.1:3000/`

### 6) Smoke test
- Register / log in
- Go to `/holds` and upload a JPG/PNG
- Go to `/wall` and upload a wall photo; you should see hold markers + a coach response

Optional API test (replace `sample.jpg`):
```bash
B64=$(base64 -i sample.jpg | tr -d '\n')
curl -s http://127.0.0.1:9000/classifier/upload \
	-H 'Content-Type: application/json' \
	-d "{\"filename\":\"sample.jpg\",\"content_type\":\"image/jpeg\",\"data\":\"$B64\"}" \
	| python3 -m json.tool
```

---
Model architecture and training instructions

architecture used:
2/3 layer approach:
1. yolo v8 detector to find holds in an image
    - crop onto the detected holds from the image, via yolov8 bounding boxes labels.
2. use a convnext classifier to classify the cropped hold images into their respective classes.
    - return the results.

additional:

3. path-finding algorithm to find a climbing path from the detected holds and their classifications.
    - needs all previously mentioned steps to be done on the image.


---
Train a model to classify climbing holds from images

Take the dataset "indoor-climbing-gym-hold-classification-dataset" from kaggle and extract the images and labels.

https://www.kaggle.com/datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/data

use "convert-to-folders.py" to convert the dataset into a folder structure that can be used by the training scripts.

to create the model.

1. train a convnext classifier model on the climbing hold images.
- run "two_phases_train.py" 
    - uses the preprocessed dataset in folder structure. called "holds_cls". must be in the same directory as the script.
    - this script will train a convnext model in two phases:
        - phase 1: train only the classifier head for x epochs.
        - phase 2: fine-tune the entire model for y epochs.

    - x, y are adjustable hyperparameters in the script.

to run inference on a single image using ONLY the convnext
validator script:
``` Bash
python predict.py -m 'path/to/your/model.pt' -i 'path/to/your/images/'
```


2. train the detector (yolov8) model from scratch on a dataset of climbing hold images.


``` Bash   
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=gpu  # use cpu if no gpu: device=cpu
```

validate the trained detector model
``` Bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640
```

inference on a raw image using ONLY the yolo detector
``` Bash
yolo detect predict model=runs/detect/train/weights/best.pt source="path/to/your/raw_folder" imgsz=640 save_txt save_conf
```

data.yaml - describes the types of things to look for in the detector model(yolo)

---

run 2 layer model on images to detect and classify holds:
``` Bash
python detect_and_classify.py -i "path/to/raw_image.jpg " -y "path/to/yolo/detector-model.pt" -c "path/to/convnext/classifier-model.pt"
```

Options:

* -i / --image: path to input image
* -y / --yolo: Path to YOLO model (default: runs/detect/train2/weights/best.pt) 
* -c / --classifier: Path to ConvNeXt model (default: best_convnext_two_phase.pt)
* --conf: YOLO confidence threshold (default: 0.25)
* --padding: Box padding fraction (default: 0.15 = 15%)
* --no-save: Skip saving visualization

Outputs annotated image with ConvNeXt predictions + confidence scores.

--- 
Optional script:
Use YOLO to crop all your YOLO-labeled images, creating a new dataset structured for classifier fine-tuning.

``` Bash
python generate_crops_for_finetuning.py
    -t 'path/to/yolo/labeled/images/' 
    -y 'path/to/yolo/detector-model' 
    -o 'path/to/output/cropped/dataset/'
```

change the paths as needed when running the script for finetuning dataset generation. "DATA_DIR = "holds_cls_finetuned""

also change this line in the "two_phases_train.py" script to use the new dataset for finetuning:
``` Python
lr=5e-5,  # lower LR for fine-tuning
```

After generating the new cropped dataset, run the "two_phases_train.py" script to finetune the convnext classifier on the new dataset.
``` Bash
python two_phase_train.py
```

---

to detect and classify holds in an image:
``` Bash
python detect_and_classify.py -i 'path/to/your/images/' -y 'path/to/yolo/detector-model' -c 'path/to/your/classifier-model'
```

---
# Requirements
- see requirements.txt

```Bash
pip install -r requirements.txt
```

---
# Web API #
- The model has a web API using FastAPI.

To turn on a web API server using FastAPI on port 9000:
``` Bash 
uvicorn main:app --reload --port 9000 
```
MAKE SURE TO ADD PYTHONPATH IF NEEDED:
``` Bash
$env:PYTHON_BIN = "C:\Users\sunna\Code\uottahacks\env\Scripts\python.exe"
```

