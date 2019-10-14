# MultiPoseNet in tensorflow (*work in progress :wrench:*)

This an implementation of [MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network](https://arxiv.org/abs/1807.04067).

## How to use this

1. Download COCO dataset.
2. Run `create_tfrecords.py`.
3. Run `train_keypoints.py`.
4. Run `train_person_detector.py`.
5. Run `train_prn.py`.
6. Run `create_pb.py`.

## Requirements
1. tensorflow 1.14
2. Pillow 6.1, opencv-python 4.1
3. numpy 1.16, scipy 1.3
4. matplotlib 3.1, tqdm 4.36
5. [pycocotools](https://github.com/cocodataset/cocoapi/)
