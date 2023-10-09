import numpy as np
import struct
import matplotlib.pyplot as plt
import os


# download the mnist dataset from http://yann.lecun.com/exdb/mnist/
# 读取要训练的数据集
def read_train_data(path=''):
    with open(os.path.join(path, 'train-images.idx3-ubyte'), 'rb') as f1:
        buf1 = f1.read()
    with open(os.path.join(path, 'train-labels.idx1-ubyte'), 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


# 读取要测试的数据集
def read_test_data(path=''):
    with open(os.path.join(path, 't10k-images.idx3-ubyte'), 'rb') as f1:
        buf1 = f1.read()
    with open(os.path.join(path, 't10k-labels.idx1-ubyte'), 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


# 得到图片数据
def get_image(buf1):
    image_index = 0
    '''
    struct.calcsize('>IIII') 是Python中用于计算二进制数据格式字符串的大小的函数调用
    它描述了一个包含四个大端字节序的无符号整数的数据结构的大小
    '''
    image_index += struct.calcsize('>IIII')
    img_num = int((len(buf1) - 16) / 784)
    im = []
    for i in range(img_num):
        temp = list(struct.unpack_from('>784B', buf1, image_index))  # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(temp)
        image_index += struct.calcsize('>784B')  # 每次增加784B
    im = np.array(im, dtype=np.float32)  # 将图片作为数据源传入numpy数组
    return im


# 得到标签数据
def get_label(buf2):
    label_index = 0
    '''
    用于计算二进制数据格式字符串的大小的函数
    '>II' 是格式字符串，它描述了要打包或解包的数据结构。
    在这个格式字符串中，> 表示使用大端字节序（big-endian），
    I 表示一个无符号整数（unsigned int）。
    '''
    label_index += struct.calcsize('>II')
    idx_num = int(len(buf2) - 8)
    labels = []
    for i in range(idx_num):
        temp = list(struct.unpack_from('>1B', buf2, label_index))
        labels.append(temp)
        label_index += 1
    labels = np.array(labels, dtype=int)  # 将标签作为数据源传入numpy数组
    return labels


# 加载训练数据
def load_train_data(path=''):
    img_buf, label_buf = read_train_data(path)
    imgs = get_image(img_buf)
    labels = get_label(label_buf)

    return imgs, labels


# 加载测试数据
def load_test_data(path=''):
    img_buf, label_buf = read_test_data(path)
    imgs = get_image(img_buf)  # 得到图片数据
    labels = get_label(label_buf)  # 得到标签数据

    return imgs, labels


if __name__ == "__main__":

    imgs, labels = load_test_data()

    for i in range(9):
        # 将整个图像窗口分为3行3列
        plt.subplot(3, 3, i + 1)
        # 设置窗口的标题
        title = u"标签对应为：" + str(labels[i])
        # 显示窗口的标题
        plt.title(title, fontproperties='SimHei')
        # 将读取的图片数据源改成28x28的大小
        img = np.array(imgs[i]).reshape((28, 28))
        # cmap参数: 为调整显示颜色 gray为黑白色，加_r取反为白黑色
        plt.imshow(img, cmap='gray')
    plt.show()
