# depth-enhanced-yolo
This is the repo that change the Yolo model so that it can have better performance handling RGBD dataset.
- train 4 rgb resolution 1088
- train 1 rgbd concatenate resolution 640
- train 2 rgb resolution 640
- train 3 dformer rgbd resolution 640 small model
- train 5 rgbd concatenate resolution 640 epoch 400
- train 6 rgb resolution 640 epoch 400
- train 7 dformer rgbd resolution 640 small model epoch 400
- train 8 dformer rgbd resolution 640 small model epoch 400 improved dataset
- train 9 rgbd concatenate resolution 640 epoch 400 improved dataset
- train 10 rgb resolution 640 epoch 400 improved dataset
- train 14 rgb-detr resolution 640 epoch 400 improved dataset (need debug)