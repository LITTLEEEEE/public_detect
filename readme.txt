github链接:https://github.com/LITTLEEEEE/public_detect.git
1. 所需环境
tensorflow2.2
keras 2.1.5
cuda10.2（如果使用gpu加速的话）
其余所需要安装的包在requirements.txt中有详细列出，可以运行命令pip install -r requirements.txt直接一键安装

2. 代码各部分说明
/font中是可视化时所使用的字体文件
/img中提供了几张测试图片
/input是为计算mAP值所用
/logs存放训练过程中产生的模型文件
/model_data中存放了初始的模型权重（init.h5）、未经过改变的训练权重（unchanged.h5）、经过修改后的模型训练权重（changed.h5）、类别说明（public.txt）、初始检测框参数（yolo_anchors.txt）
/nets下存放模型文件、损失函数文件
/results下存放mAP值相关结果图片、文件
/utils下存放一些特殊功能的实现代码，如余弦退火算法
/VOCdevkit下存放数据集
trains.py是模型训练代码
net_show.py用于显示网络结构
predict.py可以利用训练后的模型进行图片检测
get_dr_txt、get_gt_txt、get_map用于计算mAP值
kmeans_for_anchors.py用于初始检测框参数生成
video.py用于利用摄像头或者视频进行检测
voc_annotation.py、date_trans.py用于数据集预处理


3. 代码使用

如果说直接利用已有模型进行预测的话，可以在yolo.py中修改model_path和classes_path使其对应训练好的文件，然后利用predict.py或者video.py进行预测
运行predict.py，输入img/1.jpg即可，（目前已经修改好了，可以直接跑）
运行video.py，在赋予权限的情况下可以实现视频的检测

如果利用网络进行训练

将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。
将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
在训练前运行voc2yolo4.py文件生成对应的txt。
再修改根目录下的voc_annotation.py，将classes改成所需的classes。
在model_data下新建一个txt文档，文档中输入需要分的类。
运行train.py即可开始训练

（注意：不要使用中文命名，文件路径下不要有空格）

