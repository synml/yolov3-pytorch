# YOLOv3-pytorch

> A Simple PyTorch Implementation of YOLOv3.

이 프로젝트는 C로 구현된 원본 YOLOv3를 PyTorch로 재구현했습니다. YOLOv3의 학습, 테스트, demo를 지원합니다.

## Project Introduction

- Motivation
  - YOLOv3가 PyTorch에서 작동할 수 있도록 하기 위해서 만들었습니다.
- Purpose
  - YOLOv3의 train, test, demo를 PyTorch에서 수행할 수 있습니다.
- Main functions
  - 딥러닝 모델의 핵심 기능 구현 (Train, Test, Demo)
  - 최대한 YOLOv3의 논문에서 언급한 것을 그대로 구현했습니다.

## Build Status

![GitHub release (latest by date)](https://img.shields.io/github/v/release/LulinPollux/yolov3-pytorch)
![GitHub last commit](https://img.shields.io/github/last-commit/LulinPollux/yolov3-pytorch)
![GitHub](https://img.shields.io/github/license/LulinPollux/yolov3-pytorch)

## How to Install

1. Anaconda 또는 Miniconda가 설치되어 있는지 확인하세요.

2. 새 가상환경을 만들고 아래의 개발 환경 섹션에서 언급한 라이브러리를 설치하세요.

   - ```python
     conda install python=3.7 matplotlib tqdm
     ```

   - https://pytorch.org/get-started/locally/

3. 리포지토리를 클론, 포크하거나 압축파일로 코드를 다운로드하세요.

4. 데이터셋을 준비하세요.
   1. COCO: setup_coco_dataset.sh를 읽어보고 사전 준비를 한 후, 아래의 명령을 수행하세요.

      ```shell
      $ mv data/setup_coco_dataset.sh ./
      $ bash setup_coco_dataset.sh
      ```

   2. VOC
      1. [여기](http://host.robots.ox.ac.uk/pascal/VOC/)에서 데이터셋을 다운로드합니다.
      2. 아카이브 파일을 추출합니다. (VOCdevkit/VOC2007, VOCdevkit/VOC2012)
   3. voc_label.py의 docs를 읽어보고 YOLOv3에서 사용할 라벨을 만드세요.
   
5. 사전 훈련된 가중치 파일을 다운로드해서 ./weights 폴더로 옮기세요.

   1. [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   2. [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)

## How to Use, Example

1. You should check arguments of each Python file and set it according to your environment.

2. Demo

   - ```shell
     $ python demo.py
     ```

   - If you click the below picture, this page will move to the Youtube video page.
[![Youtube video](https://img.youtube.com/vi/X0LAgilivvw/maxresdefault.jpg)](https://youtu.be/X0LAgilivvw)

3. Train

   - ```shell
     $ python train.py
     ```

   - You can check the training process with the TensorBoard.

   - ```shell
     $ bash exec_tensorboard.sh
     ```

4. Test

   - ```shell
     $ python test.py
     ```

   - Test results are saved as csv files. The table below shows the test results for each method.

     - | Method                      | mAP (.50 IoU) | Inference time (ms)<sup>1</sup> |
       | :-------------------------- | ------------- | ------------------------------- |
       | YOLOv3 320 (paper)          | 51.5          | 22                              |
       | YOLOv3 320 (implementation) | 52.5          | 32                              |
       | YOLOv3 416 (paper)          | 55.3          | 29                              |
       | YOLOv3 416 (implementation) | 55.6          | 36                              |
       | YOLOv3 608 (paper)          | 57.9          | 51                              |
       | YOLOv3 608 (implementation) | 57.1          | 48                              |
       
       <sup><strong>1</strong></sup> Results vary significantly depending on the execution environment.

## API, Framework

- Python 3.7.7
- PyTorch 1.5.1
- Matplotlib
- Tensorboard 2.2.1
- Tqdm

## Develop environment

- H/W develop environment
  - Intel Core i7-9700 CPU
  - RAM 32GiB
  - Geforce RTX 2070 Super
  - Samsung SSD 970 PRO 512GB
- S/W develop environment
  - Ubuntu 20.04 LTS (Primary support)
  - Windows10 20H1 (Secondary support)
  - Miniconda
  - Cuda 10.2
- Setting up develop environment
  1. Make sure you have Anaconda or Miniconda installed.
  2. Create a new virtual environment and install the above library.
  3. Download the code to your repository as a clone, fork or ZIP file.
  4. Open the downloaded file with your IDE or text editor.
  5. Start coding~!

## Developer information and credits

- Developer
  - Clova - [Github Profile](https://github.com/clovadev)

- Credits
  - YOLOv3: An Incremental Improvement [[Paper\]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Authors' Webpage\]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation\]](https://github.com/pjreddie/darknet)
  - [How to implement a YOLOv3 object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
  - eriklindernoren's PyTorch-YOLOv3 [[Github\]](https://github.com/eriklindernoren/PyTorch-YOLOv3)

## Contribution method

1. Fork [this project](https://github.com/LulinPollux/yolov3-pytorch).
2. Create a new branch or use the master branch in the GitHub Desktop.
3. Commit the modification.
4. Push on the selected branch.
5. Please send a pull request.

## License

MIT License © Lulin

You can find more information in `LICENSE`.
