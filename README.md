
# Cube Detection + Rotation
![Screenshot](example_video.gif)

# Run

```sh
rosrun cube_detector ros_detect.py 
```


# Training

```sh
python3 train.py --img 640 --batch 3 --epochs 3 --data aug_cube.yaml  
```


# Test

```sh
python3 detect.py --source 0 --weights m640rot.pt --img 640 
```

