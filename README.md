# Training

```sh
python3 train.py --img 1920 --batch 3 --epochs 3 --data cube.yaml  
```
--weights yolov5s.pt

## Resume
```sh
python3 train.py --resume
```


# Test

```sh
python3 detect.py --source left0000.jpg --weights runs/train/exp22/weights/best.pt --img 1920 
```

