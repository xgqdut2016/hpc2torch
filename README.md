# hpc2Torch
这个仓库打算搭建一个高性能底层库的测试框架，将会针对onnx的算子编写相关的高性能kernel，作为pytorch的补充，从python端对比手写kernel和pytorch库函数的性能以及精度对比。

## src
这个文件夹下面存放的是不同算子的kernel

## test
这个文件夹存放的是不同算子的python测试脚本，其中performance.py是功能文件，用于对比性能

## run.sh
默认编译CPU端代码，运行仓库命令是：

bash run.sh

编译结束以后，可以直接做python端测试，测试softmax算子的CPU端代码命令为：

python test/test_softmax.py --device cpu

如果需要编译测试其他平台代码，比如说GPU端测试，那么修改run.sh里面的cmake ../ -DUSE_CPU=ON为 cmake ../ -DUSE_CUDA=ON，对应的测试python脚本--device cpu也修改为--device cuda

# 已知问题

## 寒武纪平台
matmul算子在f16的数据测试中精度存在巨大问题，原因不明

pool算子在f16，ndim=5的数据测试中，avgpool精度只有1e-3，但是maxpool精度正常，原因不明

clip算子在f16数据测试中精度存在巨大问题，f32正常，原因不明

## 昇腾平台
conv算子面对f32，ndim=5测试报错，但是f16，ndim=5精度正常，原因是昇腾机器torch.conv3d不支持f32

pool算子处理ndim=3的avgpool测试精度出错，昇腾平台目前缺少支持ndim=5的maxPool库函数
