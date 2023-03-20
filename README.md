<p align="center">
  <img src="https://github.com/BrasD99/Ovoxador/blob/d18f205b67df73561b9aa693a08cbf053693c096/assets/animated.gif" width="100%" height="auto" />
</p>

<h1 align="center">Ovoxador</h1>

<p>
  Welcome to <strong>Ovoxador</strong>, a cutting-edge system for training athletes like never before. Our solution leverages the power of computer vision and machine learning to process video recordings of team sport games, providing coaches and athletes with unprecedented insights into their performance.Using a combination of neural network models, our solution can detect people and sports equipment on video frames, track players, reidentify them on various tracks, get their texture, determine their pose, and acquire the field texture. These features are then presented in virtual and augmented reality, giving coaches and athletes a 360-degree view of the game and enabling them to analyze their performance in real-time. Whether you are a professional athlete striving for excellence or a coach looking to improve your team's performance, <strong>Ovoxador</strong> can help you achieve your goals. We invite you to explore our system and discover the many ways it can transform the way you train and play.
</p>

## Neural Network Models

The solution uses a variety of neural network models to solve the tasks:

- [**YOLOv8**](https://github.com/ultralytics/ultralytics) - detector for detecting people and sports equipment on video frames,
- [**DeepSort**](https://github.com/ZQPei/deep_sort_pytorch) - tracker for detecting players on video frames,
- [**TorchReId**](https://github.com/KaiyangZhou/deep-person-reid) - reidentification of players on various tracks,
- [**DensePose**](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) - getting the texture of the player on the video frame,
- [**PARE**](https://github.com/mkocabas/PARE) - determine the pose of the player on the video frame,
- [**LaMa**](https://github.com/saic-mdal/lama) - image processing and field texture acquisition.

## To-Do List

- [x] Reidentification optimization (now we are using SQLite database to store features)
- [x] Beautiful animated gif for README ;)
- [x] Multithreading support to speed up the work of modules
- [ ] Calculation of performance metrics of athletes
- [ ] GUI refinement
- [ ] VR/AR solution using Unity (will be moved to another repository)

💻 Testing on macOS 💻<br>
:warning: The project is under development! :warning:
