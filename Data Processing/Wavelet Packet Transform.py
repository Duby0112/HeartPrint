import pywt
import numpy as np
import matplotlib.pyplot as plt
import pywt.data

def wpd_plt(signal,n):
    #wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1',mode='symmetric',maxlevel=n)
 
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = signal
    for row in range(1,n+1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data
 
    #作图
    plt.figure(figsize=(18, 7))
    plt.subplot(n+1,1,1) #绘制第一个图

    plt.plot(map[1],color=plt.get_cmap('seismic')(30), linestyle='solid',marker='None', linewidth=4)
    for i in range(2,n+2):
        level_num = pow(2,i-1)  #从第二行图开始，计算上一行图的2的幂次方
        #获取每一层分解的node：['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i-1, 'freq')]  
        for j in range(1,level_num+1):
            plt.subplot(n+1,level_num,level_num*(i-1)+j)
            plt.plot(map[re[j-1]],color= plt.get_cmap('seismic')(30), linestyle='solid',marker='None', linewidth=2) #列表从0开始

data=np.load('data_2.npy')

n = 3
wpd_plt(data,n)
plt.tight_layout()
plt.show()
