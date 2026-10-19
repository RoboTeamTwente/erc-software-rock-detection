# Rock Detection Model

This repository contains the first model trained for the rock detection task.

It includes the training notebook and a ROS 2 node for real-time rock detection with a RealSense D435i camera.

## Files

- `ERC_Rock_Detection_MarsRocks.ipynb` - Model training and evaluation.
- `rock_detector_node.py` - ROS 2 rock detection node.

## Run the detector

```bash
python3 rock_detector_node.py --ros-args -p weights:=/path/to/best.pt
```

The camera must publish aligned depth data with `align_depth.enable:=true`.
