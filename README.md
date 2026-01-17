# climBright
A rock climbing hold classifier

---
to train a model to classify climbing holds from images

2 option:
1. Use a pre-trained model (ResNet50) and fine-tune it on a dataset of climbing hold images.
- run "two_phases_train.py"

2. train a yolo detector small model from scratch on a dataset of climbing hold images.
``` Bash   
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=gpu  # use cpu if no gpu: device=cpu
```

validator script:
``` Bash
python predict.py -m 'path/to/your/model.pt' -i 'path/to/your/images/'
```

data.yaml - describes the types of things to look for in the detector model(yolo)


to detect and classify holds in an image:
``` Bash
python detect_and_classify.py -i 'path/to/your/images/' -y 'path/to/yolo/detector-model' -c 'path/to/your/classifier-model'
```

# Requirements
- see requirements.txt


To turn on a web API server using FastAPI on port 9000:
``` Bash 
uvicorn main:app --reload --port 9000 
```