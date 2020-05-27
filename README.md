# SiamFC tracker for twin-beam DEP micromanipulation 

## Demo
From the root of the project, run:
```bash
python3 bin/run_video.py \
--video_path data/test.mp4 \
--model_path models/baseline-conv5_e55.pth \
--write_output False \
--visualization True \
--benchmark False
```

## Real-time application
Modify `src/config.py` with actual settings and make sure 3D_scanning is running (https://github.com/aa4cc/twinbeam-setup/blob/master/JetsonCode/3D_scanning).
Finally, run:
```bash
python3 bin/run_realtime.py
```

## Experiments

### Tracking in green and red channels

[![Visual tracking for DEP manipulation](https://img.youtube.com/vi/7sno9vlXWDo/0.jpg)](https://youtu.be/7sno9vlXWDo "Visual tracking for DEP manipulation")

### Optimal vs direct trajectory

[![Trajectory optimization for DEP manipulation](https://img.youtube.com/vi/HCItMj1XiAE/0.jpg)](https://youtu.be/HCItMj1XiAE "Trajectory optimization for DEP manipulation")

### Control algorithms

[![Control algorithms for DEP manipulation](https://img.youtube.com/vi/YZoCo-Zlh-Q/0.jpg)](https://youtu.be/YZoCo-Zlh-Q "Control algorithms for DEP manipulation")

## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
		
[2] https://github.com/StrangerZhang/SiamFC-PyTorch	

[3] https://github.com/bilylee/SiamFC-TensorFlow