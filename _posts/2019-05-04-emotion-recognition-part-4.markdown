---
layout: post
title:  "Tensorflow + FER2013 表情识别 (四)"
date:   2019-05-17 18:44:00 +0800
---

本系列博客描述如何利用Tensorflow完成人类面部表情识别CNN模型的训练与应用, 由四部分组成:
1. [模型素材库准备工作][part-1];
2. [模型的定义与训练][part-2];
3. [模型的保存与读取][part-3];
4. **在iOS应用中使用模型**;

训练环境:

|名称        |版本
|-----------|---|
|macOS      |10.13.6 ([如何配置macOS平台的GPU训练][tf-gpu-mac])
|Python     |3.6.8
|Tensorflow |1.8.0 (因兼容性问题, 未选择最新版本)
|CUDA       |[10][CUDA]
|cuDNN      |[7.4][cuDNN]

### 模型转化

[第三部分][part-3]中介绍了如何方便地保存和读取`.pb`模型. 生产环境下, 大型模型导出的`.pb`文件体积往往较大, 内含几何数量级的神经元. 对于桌面端和后端来说, 不论是算力, 功耗还是存储空间, 这些都不是问题; 但在寸土寸金, 算力有限的移动端, 庞大的模型意味着臃肿的App体积, 漫长的下载, 还有耗时耗电的运算过程.   

在移动端, 我们需要精简版的TensorFlowLite模型(`.tflite`), 下面介绍如何生成`tflite`模型:

#### 1. 冻结

首先, 使用TensorFlow内置的`freeze_graph`命令将训练过程中的checkpoint冻结为`pb`模型:

```bash
cd /path/to/project

freeze_graph \
--input_graph=path/to/graph.pbtxt \
--input_checkpoint=path/to/model.ckpt-xxx \
--input_binary=false \
--output_graph=frozen_model.pb \
--output_node_names=#tensor_name#
```

#### 2. 精简

完整的TensorFlow模型包含一些"冗余"的操作和运算层, 例如只有在训练中才会用到的操作, 移除这些不需要的部分有助于缩小模型文件的体积. 此外, 在合理范围内降低某些运算步骤的精度, 可以提高整体的运算速度. 

精简通过`optimize_for_inference`命令来完成(该命令需要事先[编译](opt_comp), 根据硬件配置, 编译耗时可能较长): 

```bash
cd /path/to/tensorflow

# 编译
bazel build tensorflow/python/tools:optimize_for_inference

# 精简
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=path/to/frozen_model.pb \
--output=path/to/optimized_model.pb \
--frozen_graph=True \
--input_names=#input_tensor_name# \
--output_names=#output_tensor_name#
```

#### 3. 转化

最后, 使用同样是TensorFlow内置的`toco`命令生成TensorFlowLite模型:

```bash
cd /path/to/project

toco \
--graph_def_file=path/to/optimized_model.pb \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=FLOAT \
--input_type=FLOAT \
--input_arrays=#input_tensor_name# \
--output_arrays=#output_tensor_name# \
--input_shapes=1,48,48,1 \
--output_file=path/to/model.tflite
```

### iOS端的应用

下面介绍如何在iOS中使用TensorFlowLite模型完成表情识别:

#### 1.安装依赖库

在Podfile中添加依赖库:

```
pod 'TensorFlowLiteSwift`
```

安装依赖库

```bash
pod repo update
pod install
```

#### 2.加载模型

```swift
#import TensorFlowLite

guard let modelPath = Bundle.main.path(forResource: "fer2013_cnn", ofType: "tflite") else {
    fatalError("tflite model not found")
}
var options = InterpreterOptions()
options.threadCount = 1
options.isErrorLoggingEnabled = true
var interpreter: Interpreter!
do {
    interpreter = try Interpreter(modelPath: modelPath, options: options)
} catch let error {
    print(error)
}
```

#### 3.使用模型

使用模型前, 先将图片尺寸调整为48*48像素, 并转成的`uint8`灰度格式, 具体操作不在此赘述.
将调整后的图片转成`Data`数据: 

```swift
extension Data {
    
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
    
    /// Creates a new buffer from CVPixelBuffer;
    ///
    /// - Parameter pixelBuffer: image in CVPixelBuffer format;
    init?(buffer_GRAY8 pixelBuffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let mutableRawPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            fatalError("unable to get pixel buffer base address")
        }
        let count = CVPixelBufferGetDataSize(pixelBuffer)
        let bufferData = Data(bytesNoCopy: mutableRawPtr, count: count, deallocator: .none)
        var grayBytes = [Float32](repeating: 0, count: 48*48)
        var index = 0
        for component in bufferData.enumerated() {
            let offset = component.offset
            // drop last 8 bytes
            if offset >= count - 8 {
                break
            }
            grayBytes[index] = Float32(component.element) / 255.0
            index += 1
        }
        self.init(copyingBufferOf: grayBytes)
    }
}
```

将`Data`传入模型, 得出计算结果:

```swift
let pixelBuffer = ...   // image buffer
var outputTensor: Tensor? = nil
do {
    try interpreter.allocateTensors()
    let inputTensor = try interpreter.input(at: 0)
    print(inputTensor)
            
    guard let grayData = data(from: pixelBuffer) else {
        fatalError("convert pixel buffer to data failed")
    }

    try interpreter.copy(grayData, toInputAt: 0)
    try interpreter.invoke()
            
    outputTensor = try interpreter.output(at: 0)
} catch let error {
    print(error)
}
print(outputTensor)
let results = [Float32](unsafeData: outputTensor!.data) ?? []
print(results)
```

输出如下:

```console
# 模型网络的入口
Tensor(
    name: "input_tensor",
    dataType: TensorFlowLite.TensorDataType.float32,
    shape: TensorFlowLite.TensorShape(rank: 4, dimensions: [1, 48, 48, 1]),
    data: 9216 bytes,
    quantizationParameters: nil
)

# 模型网络的出口
Tensor(
    name: "softmax_tensor",
    dataType: TensorFlowLite.TensorDataType.float32,
    shape: TensorFlowLite.TensorShape(rank: 2, dimensions: [1, 7]),
    data: 28 bytes,
    quantizationParameters: nil
)

# 得出的计算结果为float数组, 对应7种不同的表情以及它们的可信度(confidence)
[0.1521615, 0.08033131, 0.1378663, 0.15033567, 0.16004953, 0.1168407, 0.20241506]
```

#### 没有下一步了

[CUDA]: https://developer.nvidia.com/cuda-10.0-download-archive
[cuDNN]: https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10
[tf-gpu-mac]: https://medium.com/xplore-ai/nvidia-egpu-macos-tensorflow-gpu-the-definitive-setup-guide-to-avoid-headaches-f40e831f26ea
[fer2013]: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
[part-1]: {% post_url 2019-05-04-emotion-recognition-part-1 %}
[part-2]: {% post_url 2019-05-04-emotion-recognition-part-2 %}
[part-3]: {% post_url 2019-05-04-emotion-recognition-part-3 %}
[part-4]: {% post_url 2019-05-04-emotion-recognition-part-4 %}
[opt_comp]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py