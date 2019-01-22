#coding=utf-8
#from skimage import io,transform
import cv2
import tensorflow as tf
import numpy as np
import os


path1 = r"/home/wang/CNN/data/test"


w=100
h=100
c=1

label_class={'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'A','11':'B','12':'C','13':'D','14':'E','15':'F','16':'G','17':'H','18':'I','19':'J','20':'K','21':'L','22':'M','23':'N','24':'O','25':'P','26':'Q','27':'R','28':'S','29':'T','30':'U','31':'V','32':'W','33':'X','34':'Y','35':'Z','36':'?'}

def read_one_image(path):
    img = cv2.imread(path,0)
    img = cv2.resize(img,(w,h))
    #print(img.shape)
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    images=[]
    for imgname in os.listdir(path1):
        images.append(imgname)
        data1 = read_one_image(path1+'/'+imgname)
        #print(data1.shape)
        data1.shape+=(1,)
        data.append(data1)

    saver = tf.train.import_meta_graph(r'/home/wang/CNN/data/model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint(r'/home/wang/CNN/data/model'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    #print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    #print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    print('image    result')
    result=''
    for i in range(len(output)):
        print(str(images[i])+' '+label_class[str(output[i])])
        result+=label_class[str(output[i])]
    print('识别结果：'+result)
