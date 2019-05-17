---
layout: post
title:  "Tensorflow + FER2013 表情识别 (三)"
date:   2019-05-04 17:21:00 +0800
---

本系列教程描述如何利用Tensorflow完成人类面部表情识别CNN模型的训练与应用, 由四部分组成:
1. [模型素材库准备工作][part-1];
2. [模型的定义与训练][part-2];
3. **模型的保存与读取**;
4. [在iOS应用中使用模型][part-4];

训练环境:

|名称        |版本
|-----------|---|
|macOS      |10.13.6 ([如何配置macOS平台的GPU训练][tf-gpu-mac])
|Python     |3.6.8
|Tensorflow |1.8.0 (因兼容性问题, 未选择最新版本)
|CUDA       |[10][CUDA]
|cuDNN      |[7.4][cuDNN]

[第二部分][part-2]中阐述了如何搭建和训练模型, 整个训练过程可以分为两个阶段: 训练不足(underfit)和过度训练(overfit).

* 训练不足阶段, 模型在努力提取/分析/归纳图片的共性高维度特征, 但仍未达到能够准确判断图片内在含意的程度;
* 过度训练阶段, 模型已经掌握了部分共性特征, 但同时也提取了部分非共性特征. 这些非共性特征来自训练素材, 且只适用于训练素材, 它们会降低模型在实际应用中的准确度.

<img src="{{site.baseurl}}/images/1f7291faf1f51aa31649476837c438b46abc54e6.jpg" alt="曲线图" width="420" />

两个阶段并没有明确的分界点, 从曲线图上可以发现:

* 训练初期, 评估准确度迅速提升, 此时模型处于明显的训练不足阶段;
* 训练中期, 准确度依旧在爬升, 但是随着非共性特征的逐渐累计, 提升速度减缓;
* 训练后期, 非共性特征开始干扰模型的判断, 准确度在某个数值区间内波动, 几乎不再提升;

本文将介绍如何在训练过程中保存准确度最高的单个或多个模型, 已经如何在应用时读取它们.

### 1. 模型导出工具类

根据API, 评估配置`tf.estimator.EvalSpec`在创建时支持传入单个或多个`tf.estimator.Exporter`对象, 便于在评估结束后将当前的模型导出为`.pb`文件.
 
Tensorflow有两个现成的子类`tf.estimator.FinalExporter`和`tf.estimator.LatestExporter`, 前者在训练结束时触发, 后者在每次评估结束时触发, 但它们还不够"智能", 不能满足我们的需求. 所以, 我们需要自定义一个`HighestAccuracyExporter`:

```python
import tensorflow as tf


class HighestAccuracyExporter(tf.estimator.LatestExporter):
    _best_accuracy = 0.0

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        accuracy = eval_result['accuracy']
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            return super().export(estimator, export_path, checkpoint_path, eval_result, is_the_final_export)
```

### 2. 优化评估配置对象

我们将`HighestAccuracyExporter`植入评估配置中:

```python
# 定义待识别数据的维度, 也就是未来应用模型时载入的图片数据格式
def serving_input_receiver_fn():
    inputs = {'Pixels': tf.placeholder(tf.float32, [None, 48, 48, 1])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# 创建自定义的exporter
exporter = HighestAccuracyExporter(
    name='highest_accuracies',
    serving_input_receiver_fn=serving_input_receiver_fn
)

# 优化后的评估配置
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda : csv_input_fn([k_EVAL_CSV_PATH], BATCH_SIZE, training=False),
    steps=NUM_EVAL_DATA//BATCH_SIZE,
    throttle_secs=600,
    exporters=[exporter]
)
```

重新开始训练, 现在评估过程中5个(默认)最高准确度节点的模型将被保存下来:

```bash
...
INFO:tensorflow:Evaluation [225/256]
INFO:tensorflow:Evaluation [250/256]
INFO:tensorflow:Evaluation [256/256]
INFO:tensorflow:Finished evaluation at 2019-05-07-08:19:01
INFO:tensorflow:Saving dict for global step 2616: accuracy = 0.41350445, global_step = 2616, loss = 1.5352077
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Signatures INCLUDED in export for Classify: None
INFO:tensorflow:Signatures INCLUDED in export for Regress: None
INFO:tensorflow:Signatures INCLUDED in export for Predict: ['prediction', 'serving_default']
INFO:tensorflow:Restoring parameters from ./models/1557216965/model.ckpt-2616
INFO:tensorflow:Assets added to graph.
INFO:tensorflow:No assets to write.
INFO:tensorflow:SavedModel written to: b"./models/1557216965/export/highest_accuracies/temp-b'1557217141'/saved_model.pb"
...
```

注: 如果我们需要保存当前最新节点的模型, 也很简单, 只需调用:

```python
estimator.export_savedmodel('saved_model', serving_input_receiver_fn)
```

### 3.模型的读取

我们可以通过`tensorflow.contrib.predictor`读取保存的`.pb`模型文件:

```python
from tensorflow.contrib import predictor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

EMOTIONS  = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]

# 从测试素材库中提取一张随机图片
PRED_CSV  = 'path/to/pred.csv'
pred_df    = pd.read_csv(PRED_CSV)
random_img = pred_df.sample(1)
true_label = random_img.pop('label').values[0]
true_label = EMOTIONS[true_label]
random_img = random_img.values[0]
random_img = random_img.reshape((48,48))

# 传入模型前, 先处理图片
feed = np.reshape(random_img, (48,48,1))/255.0
feed = feed.astype(np.float32)

# 加载已保存的模型文件
MODEL_DIR = 'path/to/model'
predict_fn = predictor.from_saved_model(MODEL_DIR)

# 对图片进行判断
prediction = predict_fn({'Pixels': [feed]})
pred_label = prediction['classes'][0]
probability = prediction['probabilities'][0][pred_label]
pred_label = EMOTIONS[pred_label]

# 对比结果
plt.imshow(random_img, cmap='gray')
plt.title(true_label)
plt.xlabel('prediction: {}-{:.2f}%'.format(pred_label, probability * 100))
plt.show()
```

<img src="{{site.baseurl}}/images/7943a7728efbdac3544a9ebbef7f68f5bc586300.png" width="300" />

### 小结

本文结合实际训练场景介绍了如何保存和读取`.pb`模型, 这种格式的模型可以很方便的在桌面端和后端应用. 

### 下一步

[在iOS应用中使用模型][part-4]

[CUDA]: https://developer.nvidia.com/cuda-10.0-download-archive
[cuDNN]: https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10
[tf-gpu-mac]: https://medium.com/xplore-ai/nvidia-egpu-macos-tensorflow-gpu-the-definitive-setup-guide-to-avoid-headaches-f40e831f26ea
[fer2013]: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
[part-1]: {% post_url 2019-05-04-emotion-recognition-part-1 %}
[part-2]: {% post_url 2019-05-04-emotion-recognition-part-2 %}
[part-3]: {% post_url 2019-05-04-emotion-recognition-part-3 %}
[part-4]: {% post_url 2019-05-04-emotion-recognition-part-4 %}