---
layout: post
title:  "Tensorflow + FER2013 表情识别 (一)"
date:   2019-05-04 17:21:00 +0800
---

本系列教程描述如何利用Tensorflow完成人类面部表情识别CNN模型的训练与应用, 由四部分组成:
1. **模型素材库准备工作**;
2. [模型的定义与训练][part-2];
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

## 素材库

本教程选用[fer2013][fer2013]作为训练素材库.

#### 1. 源素材库的结构

```jupyter
import pandas as pd

csv_path = '.../path/to/fer2013.csv'
origin_df = pd.read_csv(csv_path)
origin_df.info()
origin_df.head()
```

输出如下:
```jupyter
RangeIndex: 35887 entries, 0 to 35886
Data columns (total 3 columns):
emotion    35887 non-null int64
pixels     35887 non-null object
Usage      35887 non-null object
dtypes: int64(1), object(2)
memory usage: 841.2+ KB

    emotion	pixels	                                                Usage
--------------------------------------------------------------------------------
0	0	70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...	Training
1	0	151 150 147 155 148 133 111 140 170 174 182 15...	Training
2	2	231 212 156 164 174 138 161 173 182 200 106 38...	Training
3	4	24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...	Training
4	6	4 0 0 0 0 0 0 0 0 0 0 0 3 15 23 28 48 50 58 84...	Training
```

共35887条数据, 每条分三列:
* emotion: 7种表情, `0-Angry, 1-Disgust, 2-Fear, 3-Happy, 4-Sad, 5-Surprise, 6-Neutral`
* pixels: 图片像素值字符串, 以空格分隔的2304个uint8值, 即48*48像素;
* Usage: 素材的用途, 分别为`Training`, `PrivateTest`, `PublicTest`;

随机素材展示:

```jupyter
import matplotlib.pyplot as plt
import numpy as np

emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
samples = origin_df.sample(4)
fig = plt.figure(figsize=(8,8))
for i, sample in enumerate(samples.values):
    emotion_name = emotion_names[sample[0]]
    usage = sample[-1]
    pixels = np.array(sample[1].split(' '), dtype=np.uint8)
    fig.add_subplot(2, 2, i+1)
    plt.imshow(pixels.reshape((48, 48)), cmap='gray')
    plt.title('{} - {}'.format(usage, emotion_name))
plt.show()
```

输出如下:

<img src="{{site.baseurl}}/images/743c584ad4d8dae0f3294ba7726b90781d02ec8d.png" alt="随机素材" width="420" />

#### 2. 素材分组

为了便于训练, 将全部素材按用途分为三组:

* `Training`用作训练, 共28709条;
* `PrivateTest`用作训练过程中的周期性评估, 共3589条;
* `PublicTest`用作最终的准确度检测, 共3589条;

```python
usages = origin_df['Usage'].unique()
print(usages)
usage_grps = origin_df.groupby('Usage').groups
print(usage_grps)
```

输出如下:
```jupyter
['Training' 'PublicTest' 'PrivateTest']

{'PrivateTest': Int64Index([32298, 32299, 32300, 32301, 32302, 32303, 32304, 32305, 32306,
            32307,
            ...
            35877, 35878, 35879, 35880, 35881, 35882, 35883, 35884, 35885,
            35886],
           dtype='int64', length=3589),
 'PublicTest': Int64Index([28709, 28710, 28711, 28712, 28713, 28714, 28715, 28716, 28717,
            28718,
            ...
            32288, 32289, 32290, 32291, 32292, 32293, 32294, 32295, 32296,
            32297],
           dtype='int64', length=3589),
 'Training': Int64Index([    0,     1,     2,     3,     4,     5,     6,     7,     8,
                9,
            ...
            28699, 28700, 28701, 28702, 28703, 28704, 28705, 28706, 28707,
            28708],
           dtype='int64', length=28709)}
```

分别保存在独立的csv文件中(确保训练中的模型不会"偷看"用于评估的图片).

```python
# 按用途分组并保存至csv;
def separate_by_usage():
    grp_names = ['Training', 'PrivateTest', 'PublicTest']
    df_names = ['train', 'eval', 'pred']
    for i,grp_name in enumerate(grp_names):
        grp = usage_grps[grp_name]
        grp_df = origin_df[grp[0]:grp[-1]+1]
        grp_df.pop('Usage')
        grp_df.info()
        print()
        grp_df.to_csv('{}.csv'.format(df_names[i]), index=False)
        

# 执行分组
separate_by_usage()
```

输出如下:

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28709 entries, 0 to 28708
Data columns (total 2 columns):
emotion    28709 non-null int64
pixels     28709 non-null object
dtypes: int64(1), object(1)
memory usage: 448.7+ KB

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3589 entries, 32298 to 35886
Data columns (total 2 columns):
emotion    3589 non-null int64
pixels     3589 non-null object
dtypes: int64(1), object(1)
memory usage: 56.2+ KB

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3589 entries, 28709 to 32297
Data columns (total 2 columns):
emotion    3589 non-null int64
pixels     3589 non-null object
dtypes: int64(1), object(1)
memory usage: 56.2+ KB
```

#### 3. 训练素材分割

训练数据多达28709条, 文件尺寸240+M, 读取如此大的csv文件还是比较耗时的, 我们将进一步切割`train.csv`为8个独立的csv, 好处是:

* 单个csv文件只有30M左右, 大大缩短读取的耗时;
* 利用多核心CPU的多线程技术完成多csv文件的同步读取;
* 通过随机打乱文件的读取顺序, 实现每条素材的随机排列(第二部分将提及), 有助于提高训练效果;

```python
# 训练素材均等分割函数
def sharding_train_data(num_shards=8):
    train_df = pd.read_csv('train.csv')
    n_data_per_shard = train_df.shape[0]//num_shards
    for i in range(num_shards):
        idx_start = n_data_per_shard * i
        if i < num_shards - 1:
            idx_end = idx_start + n_data_per_shard
        else:
            idx_end = train_df.shape[0]-1
        print('sharding train df in range: [{},{}]'.format(idx_start, idx_end))
        shard_df = train_df[idx_start:idx_end]
        shard_df.to_csv('train_{}.csv'.format(i), index=False)
        

# 进一步分割训练素材库(当前目录下将生成: train_0/1/2/3/4/5/6/7.csv)
sharding_train_data()
```

输出如下:

```bash
sharding train df in range: [0,3588]
sharding train df in range: [3588,7176]
sharding train df in range: [7176,10764]
sharding train df in range: [10764,14352]
sharding train df in range: [14352,17940]
sharding train df in range: [17940,21528]
sharding train df in range: [21528,25116]
sharding train df in range: [25116,28708]
```

注: 可以根据CPU的核心/线程数自定义切割后的csv文件数量(例如: 32核心/64线程的CPU可以考虑分割成64个csv文件, 虽然有点多).

### 小结

本文主要介绍了fer2013素材库, 并对源csv文件进行了分组/切割等预处理.

对于CNN卷积网络模型的训练来说, 素材库优化是一个非常重要的环节, 合理的结构能提高训练效率, 降低耗时, 加快模型的迭代. 在使用GPU执行训练时, 这一点尤为明显. 通常状况下, GPU运算的耗时远低于CPU加载素材的耗时, 提高素材加载速度, 不但能有效缓解GPU的饥饿状态, 更能直接缩短训练总时长. 

### 下一步

[模型的定义与训练][part-2]
    
[CUDA]: https://developer.nvidia.com/cuda-10.0-download-archive
[cuDNN]: https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10
[tf-gpu-mac]: https://medium.com/xplore-ai/nvidia-egpu-macos-tensorflow-gpu-the-definitive-setup-guide-to-avoid-headaches-f40e831f26ea
[fer2013]: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
[part-1]: {% post_url 2019-05-04-emotion-recognition-part-1 %}
[part-2]: {% post_url 2019-05-04-emotion-recognition-part-2 %}
[part-3]: {% post_url 2019-05-04-emotion-recognition-part-3 %}
[part-4]: {% post_url 2019-05-04-emotion-recognition-part-4 %}