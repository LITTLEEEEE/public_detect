# -*- coding:utf-8 -*-
import os
if __name__ == "__main__":
    rootdir = '/Users/wzh/Documents/Code/PyCharm_Project/cv_hw_dateset/1'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    num = 1498
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        dirname, filename = os.path.split(path)
        newname = dirname+'/'+str(num)+'.jpg'
        os.rename(path,newname)
        print (path+"====>"+newname)
        num+=1

    print(len(list))