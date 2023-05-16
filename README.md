# hpml-2023
# Project Description

**Model 1: Facial Expression Recognition with EfficientNet**
We profiled the model using pytorch profiler, and tried different approaches to improve the performance, including optimizer selecting, Multi-threaded Data Prefetching and number of worker tuning, mix precision training


**Model 2: Facial Expression Recognition with EfficientNet**

We research on model parallelism with PyTorch DataParallel. We also discussed relationship between scaling efficiency and batch size, and relationship between convergence and batch size.


# Repository and code structure

**Model 1: Facial Expression Recognition with EfficientNet**  
Download Dataset in input directory and arrange as below.  
EfficientNet  
--efficientNet-src  
----input  
------Train  
------Angry  
------Disgust  
------Fear  
------Happy  
------Neutral  
------Sad  
------Surprise  
----test_images  
------Angry  
------Disgust  
------Fear  
------Happy  
------Neutral  
------Sad  
------Surprise  
----outputs  

**Model 2: Facial Expression Recognition with EfficientNet**
#### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:
   
   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
    ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            `-- pose_mpii

   ```
  
#### Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

# How to execute

**Model 1: Facial Expression Recognition with EfficientNet**
Train with different hyperparameters (learning rate, batch size, optimizer, number of data workers)

```sh
python train.py --cuda --epochs 30 --pretrained --epochs {epoch_num} \
--batch_size {batch_size} --optimizer {optimizer} --num_workers {num_workers} \
--learning_rate {learning_rate}
```

Train with mix-precision

```sh
python train-mix.py --cuda --epochs 30 --pretrained --epochs {epoch_num} \
--batch_size {batch_size} --optimizer {optimizer} --num_workers {num_workers} \
--learning_rate {learning_rate}
```

Profile using pyTorch profiler

```sh
python train-profile.py --cuda --epochs 30 --pretrained --epochs {epoch_num} \
--batch_size {batch_size} --optimizer {optimizer} --num_workers {num_workers} \
--learning_rate {learning_rate}
```

**Model 2: Facial Expression Recognition with EfficientNet**

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Finetuning on COCO train2017 dataset

```
python tools/train.py \
                --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_finetune_gpu{gpu numbers}.yaml \
OUTPUT_DIR output-gpu{gpu numbers}-bs{batch size number}\
  LOG_DIR log-gpu{gpu numbers}-bs{batch size number}\
  TRAIN.END_EPOCH {epoch number} \
  TRAIN.BATCH_SIZE_PER_GPU {batch size number}
```
#### Visualizing predictions on COCO val

```
python visualization/plot_coco.py \
    --prediction output/coco/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json \
    --save-path visualization/results

```

# Results and observations

**Model 1: Facial Expression Recognition with EfficientNet**

* Optimizer
  ![optimizer](https://github.com/xiaoxuandong/hpml-2023/assets/51432551/10d4a509-8edd-4b0a-aee0-6936818f537f)
  
  * From the chart, Adam had the best performance and achieve rapid convergence.
* Multi-threaded Data Prefetching
  ![worker](https://github.com/xiaoxuandong/hpml-2023/assets/51432551/d60e7115-dcb9-43e1-8466-4b6fde3d028b)
  * As the worker number increasing, the ratio of data loading time first decreases and then increases.
  * The optimal number of worker is 8.
* PyTorch Profiler
  
  * Using profiler to analyze execution time
  <img width="995" alt="Screen Shot 2023-05-15 at 13 45 11" src="https://github.com/xiaoxuandong/hpml-2023/assets/51432551/c18363b2-0ffe-4119-9907-a2240dabad17">

    * Most of time was spent on convolution operations.
  * Using profiler to analyze memory consumption
<img width="909" alt="Screen Shot 2023-05-15 at 13 45 20" src="https://github.com/xiaoxuandong/hpml-2023/assets/51432551/c6abcb17-4194-4ec6-bbf9-301be8a64d6c">


* Mix Precision
  <img width="1161" alt="Screen Shot 2023-05-15 at 13 43 26" src="https://github.com/xiaoxuandong/hpml-2023/assets/51432551/0f0a1be8-9d96-4f20-a9c7-1e8de6ce1787">
  * The average computing time per epoch decreases significantly after applying mix precision

**Model 2: Body Pose Estimation with HRNet**

* Model Parallelism with PyTorch DataParallel

![3bs-time](https://github.com/xiaoxuandong/hpml-2023/assets/51432551/50e6762f-f976-4fab-b4ab-2d442e1dd86a)

  
  * For a fixed GPU number, model with larger batch size uses less time per epoch.
  * For a fixed batch size, training with more GPUs uses less time per epoch
* Scaling efficiency w.r.t. Batch size
  
  ![3scaling](https://github.com/xiaoxuandong/hpml-2023/assets/51432551/c8a4fe74-2d41-4701-a361-35735e356171)

  
  * Batch size 64 gives the optimal Scaling Efficiency
  * None of the three models achieved the ideal scaling.
* Batch size and convergence
  <img width="436" alt="Screen Shot 2023-05-15 at 13 44 59" src="https://github.com/xiaoxuandong/hpml-2023/assets/51432551/fd3576f9-be0c-4382-8ad7-4a77b547135f">

  * Large batch size can slow down convergence
