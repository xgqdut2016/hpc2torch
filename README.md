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

## 英伟达平台
编译cutlass相关代码需要选择合适的sm，另外对于cutlass仓库，需要把tools/util/include/cutlass/util目录移动到include/cutlass/下面
## 寒武纪平台
matmul算子在f16的数据测试中精度存在巨大问题，原因不明

pool算子在f16，ndim=5的数据测试中，avgpool精度只有1e-3，但是maxpool精度正常，原因不明

clip算子在f16数据测试中精度存在巨大问题，f32正常，原因不明

layernorm算子cnnl不支持axis=0，手写算子处理axis=0,f16时必须控制规模，规模太大会导致精度无法对齐

寒武纪reduce算子处理sum,prod时，规模太大也会带来精度问题，应该是累积误差导致

## 昇腾平台
conv算子面对f32，ndim=5测试报错，但是f16，ndim=5精度正常，原因是昇腾机器torch.conv3d不支持f32，convTranspose有类似问题

pool算子处理ndim=3的avgpool测试精度出错，对于三维向量必须手动填充为四维向量才能处理，昇腾平台目前缺少支持ndim=5的maxPool库函数

昇腾调库softmax处理f16数据精度不足

昇腾reduce算子min,prod仅支持针对某一个维度做规约，不支持同时处理多个axis

昇腾clip不管input是f16还是f32，传入的min,max都必须是f32，否则结果报错

昇腾matmul算子对于高维矩阵乘法不支持alpha, beta参数，并且对于大规模矩阵存在精度问题，怀疑是规模太大造成的误差累积

## 太初平台
causal_softmax算子必须在计算结束以后加入打印信息才能保证测试通过，如果不做打印会报错segmentation fault (core dumped)

random_sample算子多个测试也会报错segmentation fault (core dumped)，而且performance.TecoProfile调用也会导致问题，原因不详

如果在python端计算以及数值对比的时候提前把tensor转移到CPU，这样可以避免上面的问题，原因不详

## 算子定义
batchnorm算子有一个momentum参数，onnx默认值是0.9，torch默认是0.1

slice算子onnx定义中有一个optional参数axis指定切片维度，但是寒武纪和昇腾库函数不支持这个参数
