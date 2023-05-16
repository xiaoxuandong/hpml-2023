# hpml-2023
# Project Description

**Model 1: Facial Expression Recognition with EfficientNet**
We profiled the model using pytorch profiler, and tried different approaches to improve the performance, including optimizer selecting, Multi-threaded Data Prefetching and number of worker tuning, mix precision training


**Model 2: Facial Expression Recognition with EfficientNet**

We research on model parallelism with PyTorch DataParallel. We also discussed relationship between scaling efficiency and batch size, and relationship between convergence and batch size.


# Repository and code structure

**Model 1: Facial Expression Recognition with EfficientNet**  
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
