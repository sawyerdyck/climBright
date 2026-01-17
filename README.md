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

run with json output:
``` Bash
python detect_and_classify.py -i 'path/to/your/images/' -y 'path/to/yolo/detector-model' -c 'path/to/your/classifier-model' --json-out 'path/to/save/detections.json'
```

# Requirements
- see requirements.txt

to map the wall and find all possible routes from the json output of detect_and_classify.py:
``` Bash
 python .\map_wall.py `
 -i "C:\Users\sunna\Downloads\altitude2.jpg" ` 
 --holds .\detections.json `
 --homography .\save\.npy `                                    
 --overlay-out outputs/routes.png `
 --paths-out outputs/routes.json
```

First pass (saves detections JSON)
``` Bash
python detect_and_classify.py -i holds_cls/real_val/wall-with-bunch-of-holds.png --json-out outputs/detections.json
```

Capture wall corners and persist homography (only once per wall):
``` Bash
python map_wall.py -i holds_cls/real_val/wall-with-bunch-of-holds.png --holds outputs/detections.json --save-homography saved.npy --overlay-out outputs/routes.png --paths-out outputs/routes.json
```

Re-run later with existing homography(no need to capture corners again):
``` Bash
python map_wall.py -i holds_cls/real_val/wall-with-bunch-of-holds.png --holds outputs/detections.json --homography saved.npy --overlay-out outputs/routes.png --paths-out outputs/routes.json
```