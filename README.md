![Alt text](https://github.com/BrasD99/Ovoxador/blob/82c516abcdaba5af9768385d75d40fb4081377b2/assets/logo.png)

Introducing a new system for training athletes. it processes video recordings of a team sport game from various cameras, combines data and presents them in virtual and augmented reality.

The solution uses a variety of neural network models to solve the tasks:
- [**YOLOv8**](https://github.com/ultralytics/ultralytics) - detector for detecting people and sports equipment on video frames,
- [**DeepSort**](https://github.com/ZQPei/deep_sort_pytorch) - tracker for detecting players on video frames,
- [**TorchReId**](https://github.com/KaiyangZhou/deep-person-reid) - reidentification of players on various tracks,
- [**DensePose**](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) - getting the texture of the player on the video frame,
- [**MediaPipe Pose (BlazePose)**](https://github.com/google/mediapipe) - determine the pose of the player on the video frame,
- [**Lama**](https://github.com/saic-mdal/lama) - image processing and field texture acquisition.

:warning: The project is under development! :warning:
