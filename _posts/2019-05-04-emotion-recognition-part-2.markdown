---
layout: post
title:  "Tensorflow + FER2013 表情识别 (二)"
date:   2019-05-17 18:44:00 +0800
---

本系列博客描述如何利用Tensorflow完成人类面部表情识别CNN模型的训练与应用, 由四部分组成:
1. [模型素材库准备工作][part-1];
2. **模型的定义与训练**;
3. [模型的保存与读取][part-3];
4. [在iOS应用中使用模型][part-4];

训练环境:

|名称        |版本
|-----------|---|
|macOS      |10.13.6 ([如何配置macOS平台的GPU训练][tf-gpu-mac])
|Python     |3.6.8
|Tensorflow |1.8.0 (因兼容性问题, 未选择最新版本)
|CUDA       |[10][CUDA]
|cuDNN      |[7.4][cuDNN]


在[第一部分][part-1]中, 我们完成了[FER2013][fer2013]素材库的预处理(分组/切割). 本文介绍如何搭建CNN模型, 并通过[Tensorflow][tf]的[Estimator API][est-api]进行训练. 

### 0. 流程简介

* 首先, 定义模型函数`cnn_model_fn()`, 完成模型的搭建, 配置不同状态(训练/评估/应用)下模型的行为;
* 以`cnn_model_fn()`为参数, 创建`tf.estimator.Estimator`识别器对象;
* 定义素材加载函数`csv_input_fn()`, 基于此函数, 分别创建训练/评估的配置`tf.estimator.TrainSpec`和`tf.estimator.EvalSpec`;
* 调用`tf.estimator.train_and_evaluate()`, 将识别器及素材配置传入, 实现模型的训练与周期性评估. 

下面详细介绍每一个步骤;

### 1.模型函数

我们的示例模型由以下层级组成:

* Convolution部分
    * input: 4维数组(?, 48, 48, 1), 分别代表batch_size(可自定义), 高48像素, 宽48像素, 单通道(灰度), 命名为`input_tensor`;
    * bn-0: Batch Normalization层, `tf.layers.batch_normalization`;
    * conv2d-1: 滤镜数为32, 滤镜尺寸[3,3], 边缘填充;
    * avg_pool-1: pool尺寸[2,2], 步长为2, 随机drop率为0.1;
    * bn-1: Batch Normalization层, `tf.layers.batch_normalization`;
    * conv2d-2: 滤镜数为64, 滤镜尺寸[3,3], 边缘填充;
    * max_pool-1: pool尺寸[2,2], 步长为2, 随机drop率为0.1;
    * bn-2: Batch Normalization层, `tf.layers.batch_normalization`;
    * conv2d-3: 滤镜数为128, 滤镜尺寸[3,3], 边缘填充;
    * max_pool-2: pool尺寸[2,2], 步长为2, 随机drop率为0.1;
* Dense部分
    * dense-1: 神经元数256, 随机drop率为0.4;
    * bn-3: Batch Normalization层, `tf.layers.batch_normalization`;
    * dense-2: 神经元数128, 随机drop率为0.5;
    * output: 单维数组, 神经元数7, 代表7种不同的表情;

创建`cnn_model_fn()`函数:

```python
import tensorflow as tf

# Define CNN
def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # input
    net = tf.placeholder_with_default(features['Pixels'], (None, 48, 48, 1), name='input_tensor')

    # bn-0
    net = tf.layers.batch_normalization(
        inputs=net,
        training=mode == tf.estimator.ModeKeys.TRAIN,
    )

    # conv2d-1
    net = tf.layers.conv2d(
        inputs=net,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )
    
    # avg_p2d-1
    net = tf.layers.average_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2
    )

    # bn-1
    net = tf.layers.batch_normalization(
        inputs=net,
        training=mode == tf.estimator.ModeKeys.TRAIN,
    )

    # conv2d-2
    net = tf.layers.conv2d(
        inputs=net,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )
    
    # max_p2d-1
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2
    )

    # bn-2
    net = tf.layers.batch_normalization(
        inputs=net,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # conv2d-3
    net = tf.layers.conv2d(
        inputs=net,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    # max_p2d-2
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2
    )

    # CONV2D -> DENSE
    net = tf.reshape(net, [-1, 6 * 6 * 128])

    # dense-1
    net = tf.layers.dense(
        inputs=net,
        units=256,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation=tf.nn.relu
    )

    # bn-3
    net = tf.layers.batch_normalization(
        inputs=net,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # dropout-1
    net = tf.layers.dropout(
        inputs=net,
        rate=0.4,
        seed=random.randint(1, 100),
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # dense-2
    net = tf.layers.dense(
        inputs=net,
        units=128,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation=tf.nn.relu
    )

    # Logits Layer
    logits = tf.layers.dense(
        inputs=net,
        units=params['num_classes']
    )

    # Generate Predictions
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # In prediction:
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            export_outputs=export_outputs,
            predictions=predictions
        )

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Calculate Accuracy
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    # Add evaluation metrics
    eval_metric_ops = {'accuracy': accuracy}  # accuracy metric for eval graph

    # In evaluation:
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
        )

    # Add training metrics
    tf.summary.scalar('accuracy_t', accuracy[1])  # accuracy metric for train graph

    # In training:
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate']
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
```

### 2. Estimator

创建`tf.estimator.Estimator`对象

```pythonstub
MODEL_DIR = './model/'

classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=MODEL_DIR
)
```

### 3.素材读取函数

[第一部分][part-1]中提到, [FER2013][fer2013]素材库中的图片是以字符串的形式保存, 由2304(48*48像素)个整型数以空格分隔而成. 我们首先定义处理单条素材的函数:

```python
# 单条素材含两列数据, emotion(整型数), pixels(字符串)
# tensorflow将以record_defaults作为数据类型的模板, 从csv文件的每一行中提取数据;
record_defaults = [[0], ['']]

def parse_line(line):
    """parse single csv row into features/label tuple required by dataset feeding"""
    fields = tf.decode_csv(line, record_defaults=record_defaults)
    label  = fields.pop(0)
    fields = tf.string_split(fields, delimiter=' ')
    fields = fields.values
    fields = tf.string_to_number(fields)
    fields = tf.reshape(fields, [48, 48, 1])
    fields = tf.divide(fields, 255.0)
    fields = augment_color(fields)
    features = {
        'Pixels': fields
    }
    return features, label
```

定义素材加载函数`csv_input_fn()`, 从单个或多个csv文件中读取指定数量的素材, 通过`parse_line`函数进行批量处理(字符串->多为数组), 以`tensorflow.data.Dataset`的形式返回:

```python
def csv_input_fn(csv_paths, batch_size, training=True):
    assert(len(csv_paths) > 0)
    if len(csv_paths) > 1:
        # multiple csv files;
        n_files = len(csv_paths)
        shuffle(csv_paths)
        dataset = (tf.data.Dataset
                   .from_tensor_slices(csv_paths)
                   .interleave(lambda x: 
                               tf.data.TextLineDataset(x).skip(1)
                               .map(parse_line, num_parallel_calls=k_NUM_THREADS),
                               cycle_length=n_files, block_length=batch_size//n_files))
    else:
        # single csv file;
        dataset = tf.data.TextLineDataset(csv_paths[0]).skip(1)
        dataset = dataset.map(parse_line, num_parallel_calls=k_NUM_THREADS)
        
    if training:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size, count=1))
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(batch_size).cache()
        return dataset
    else:
        return dataset.batch(batch_size).prefetch(batch_size)
```

第一个参数`csv_paths`支持同时传入多个csv文件路径, CPU将开启数个线程, 同时从多个csv文件中提取素材, 直到数量达到`batch_size`为止. 这个
函数将同时用于加载训练和评估素材, 所以设定第三个boolean参数`training`, 区分当前加载素材的目的. 注意, 评估时素材无须作打乱处理.

在[第一部分][part-1]中我们已经对训练素材作了分割处理, 所以此处可以将分割的8个csv文件名传入`csv_paths`.

### 4.训练/评估素材配置

训练素材配置:

```python
k_TRAIN_CSV_PATHS = ['./train_{}.csv'.format(i) for i in range(8)]
BATCH_SIZE = 128
MAX_STEP = 20_000

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda : csv_input_fn(k_TRAIN_CSV_PATHS, BATCH_SIZE),
    max_steps=MAX_STEP
)
```

评估素材配置:

```python
import pandas as pd

k_EVAL_CSV_PATH = './eval.csv'
eval_df = pd.read_csv(k_EVAL_CSV_PATH)
NUM_EVAL_DATA = eval_df.shape[0]

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda : csv_input_fn([k_EVAL_CSV_PATH], BATCH_SIZE, training=False),
    steps=NUM_EVAL_DATA//BATCH_SIZE,
    throttle_secs=600,
)
```

### 5.开始训练

素材库就绪, 模型搭建完毕, 开始训练:

```python
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
```

输出如下:

```jupyter
INFO:tensorflow:Running training and evaluation locally (non-distributed).
INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after 600 secs (eval_spec.throttle_secs) or training is finished.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into ./models/1557216965/model.ckpt.
INFO:tensorflow:loss = 3.0442383, step = 0
INFO:tensorflow:global_step/sec: 7.853
INFO:tensorflow:loss = 2.3549774, step = 100 (15.136 sec)
INFO:tensorflow:global_step/sec: 7.1937
INFO:tensorflow:loss = 2.2652185, step = 200 (16.916 sec)
INFO:tensorflow:Saving checkpoints for 218 into ./models/1557216965/model.ckpt.
INFO:tensorflow:Loss for final step: 2.2351384.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-05-07-08:16:17
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./models/1557216965/model.ckpt-218
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Evaluation [1/28]
...
INFO:tensorflow:Evaluation [28/28]
INFO:tensorflow:Finished evaluation at 2019-05-07-08:16:22
INFO:tensorflow:Saving dict for global step 218: accuracy = 0.11941964, global_step = 218, loss = 1.9582163
```

注:
* 使用GPU来训练CNN模型更加高效, 多核心/线程CPU能缩短素材准备的时间;
* 训练过程中的周期性评估间隔可以视具体情况而定: 
  * 大型素材库完成单个epoch耗时较长, 可以指定一个合理的间隔, 单循环内评估多次;
  * 小型素材库可以在每个epoch结束时进行一次评估;
* 周期性评估有助于及时观察模型的训练效果, 配合TensorBoard使用更加直观;

### 小结

本文介绍了CNN模型的搭建, 训练/评估的素材加载配置, 以及如何训练模型.

### 下一步

[模型的保存与读取][part-3]

[CUDA]:       https://developer.nvidia.com/cuda-10.0-download-archive
[cuDNN]:      https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10
[tf]:         https://www.tensorflow.org/
[est-api]:    https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/estimator
[estimator]:  https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/estimator/Estimator
[tf-gpu-mac]: https://medium.com/xplore-ai/nvidia-egpu-macos-tensorflow-gpu-the-definitive-setup-guide-to-avoid-headaches-f40e831f26ea
[fer2013]:    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
[part-1]:     {% post_url 2019-05-04-emotion-recognition-part-1 %}
[part-2]:     {% post_url 2019-05-04-emotion-recognition-part-2 %}
[part-3]:     {% post_url 2019-05-04-emotion-recognition-part-3 %}
[part-4]:     {% post_url 2019-05-04-emotion-recognition-part-4 %}