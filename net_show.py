# 该部分代码用于看网络结构，并非测试代码

from nets.yolo4 import yolo_body
from tensorflow.keras.layers import Input

inputs = Input([416,416,3])
model = yolo_body(inputs,3,3)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name) 
