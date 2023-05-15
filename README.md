# hpml-2023
# Project Description

**Model 1: Facial Expression Recognition with EfficientNet**
We profiled the model using pytorch profiler, and tried different approaches to improve the performance, including optimizer selecting, Multi-threaded Data Prefetching and number of worker tuning, mix precision training


**Model 2: Facial Expression Recognition with EfficientNet**

We research on model parallelism with PyTorch DataParallel. We also discussed relationship between scaling efficiency and batch size, and relationship between convergence and batch size.


# Repository and code structure

**Model 1: Facial Expression Recognition with EfficientNet**



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

```sh
python train-profile.py --cuda --epochs 30 --pretrained --epochs {epoch_num} \
--batch_size {batch_size} --optimizer {optimizer} --num_workers {num_workers} \
--learning_rate {learning_rate}
```

# Results and observations

**Model 1: Facial Expression Recognition with EfficientNet**

* Optimizer
  
  * From the chart, Adam had the best performance and achieve rapid convergence.
* Multi-threaded Data Prefetching
  
  * As the worker number increasing, the ratio of data loading time first decreases and then increases.
  * The optimal number of worker is 8.
* PyTorch Profiler
  
  * Using profiler to analyze execution time
    * Most of time was spent on convolution operations.
  * Using profiler to analyze memory consumption
* Mix Precision
  
  * The average computing time per epoch decreases significantly after applying mix precision

**Model 2: Body Pose Estimation with HRNet**

* Model Parallelism with PyTorch DataParallel
  
  * For a fixed GPU number, model with larger batch size uses less time per epoch.
  * For a fixed batch size, training with more GPUs uses less time per epoch
* Scaling efficiency w.r.t. Batch size
  
  * Batch size 64 gives the optimal Scaling Efficiency
  * None of the three models achieved the ideal scaling.
* Batch size and convergence
  
  * Large batch size can slow down convergence
