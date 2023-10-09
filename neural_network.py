import numpy as np
import os
import dataloader as dl  # 导入数据加载器模块
import random
import argparse  # 导入命令行参数解析模块
import matplotlib.pyplot as plt

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
# --data_path 的默认参数为 ’./‘ 即当前路径
parser.add_argument('--data_path', type=str, default='./')
# 解析参数
args = parser.parse_args()
# 加载参数
data_path = args.data_path


# 激活函数
class Sigmoid(object):
    def __init__(self):
        self.gradient = []

    def forward(self, x):
        """
        Sigmoid激活函数的前向传播函数。
        Args:
            x (numpy.ndarray): 输入数据。
        Returns:
            numpy.ndarray: 经过Sigmoid激活后的数据。
        """
        self.gradient = x * (1.0 - x)
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self):
        """
        Sigmoid激活函数的反向传播函数。

        Returns:
            numpy.ndarray: 梯度。
        """
        return self.gradient


# ReLU激活函数
class ReLU(object):

    def __init__(self):
        self.gradient = []

    def forward(self, input_data):
        """
        ReLU激活函数的前向传播函数。
        Args:
            input_data (numpy.ndarray): 输入数据。
        Returns:
            numpy.ndarray: 经过ReLU激活后的数据。
        """
        extend_input = input_data
        self.gradient = np.where(input_data >= 0, 1, 0.001)
        self.gradient = self.gradient[:, None]
        input_data[input_data < 0] = 0.01 * input_data[input_data < 0]
        return input_data

    def backward(self):
        """
        ReLU激活函数的反向传播函数。
        Returns:
            numpy.ndarray: 梯度。
        """
        return self.gradient


def softmax(input_data):
    """
    softmax函数用于将输入数据转化为概率分布。
    Args:
        input_data (numpy.ndarray): 输入数据。
    Returns:
        numpy.ndarray: 经过Softmax处理后的概率分布。
    """
    # 减去最大值防止softmax上下溢出
    input_max = np.max(input_data)
    input_data -= input_max
    input_data = np.exp(input_data)
    exp_sum = np.sum(input_data)
    input_data /= exp_sum
    return input_data


# 全连接
class FullyConnectedLayer(object):

    def __init__(self, input_size, output_size, learning_rate=0.01):
        self._w = np.random.randn(input_size * output_size) / np.sqrt(input_size * output_size)
        self._w = np.reshape(self._w, (input_size, output_size))
        b = np.zeros((1, output_size), dtype=np.float32)
        self._w = np.concatenate((self._w, b), axis=0)
        self._w = self._w.astype(np.float32)
        # self._w = np.ones((input_size + 1, output_size), dtype=np.float32)
        self.lr = learning_rate
        self.gradient = np.zeros((input_size + 1, output_size), dtype=np.float32)
        self.w_gradient = []
        self.input = []

    def forward(self, input_data):
        """
        全连接层的前向传播函数。
        Args:
            input_data (numpy.ndarray): 输入数据。
        Returns:
            numpy.ndarray: 前向传播结果。
        """
        # 将b加入w矩阵
        input_data = np.append(input_data, [1.0], axis=0)
        input_data = input_data.astype(np.float32)
        # 计算线性乘积
        output_data = np.dot(input_data.T, self._w)
        # 保存输入数据以计算梯度
        self.input = input_data

        # 更新梯度
        self.gradient = self._w
        self.w_gradient = input_data

        return output_data

    def backward(self):
        """
        全连接层的反向传播函数。
        Returns:
            numpy.ndarray: 梯度。
        """
        return self._w[:-1, :]

    def update(self, delta_grad):
        """
        更新权重参数的函数。
        Args:
            delta_grad (numpy.ndarray): 更新的梯度。
        """
        self.input = self.input[:, None]
        self._w -= self.lr * np.matmul(self.input, delta_grad)

    def get_w(self):
        """
        获取权重参数的函数。
        Returns:
            numpy.ndarray: 权重参数。
        """
        return self._w

    def set_w(self, w):
        """
        设置权重参数的函数。
        Args:
            w (numpy.ndarray): 要设置的权重参数。
        """
        self._w = w


# 交叉熵损失函数
class CrossEntropyWithLogit(object):

    def __init__(self):
        self.gradient = []

    def calculate_loss(self, input_data, y_gt):
        """
        计算交叉熵损失函数。
        Args:
            input_data (numpy.ndarray): 输入数据。
            y_gt (numpy.ndarray): 真实标签。
        Returns:
            float: 损失值。
        """
        input_data = softmax(input_data)
        # 交叉熵公式 -sum(yi*logP(i))
        loss = -np.sum(y_gt * np.log(input_data + 1e-5))
        # 计算梯度
        self.gradient = input_data - y_gt

        return loss

    def predict(self, input_data):
        """
        预测函数，返回预测的类别。
        Args:
            input_data (numpy.ndarray): 输入数据。
        Returns:
            int: 预测的类别。
        """
        # input_data = softmax(input_data)
        return np.argmax(input_data)

    def backward(self):
        """
        交叉熵损失函数的反向传播函数。
        Returns:
            numpy.ndarray: 梯度。
        """
        return self.gradient.T


# MNIST神经网络模型
class MNISTNet(object):

    def __init__(self):
        self.linear_layer1 = FullyConnectedLayer(28 * 28, 128)
        self.linear_layer2 = FullyConnectedLayer(128, 10)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.loss = CrossEntropyWithLogit()

    def train(self, x, y):
        """
        训练函数，包括前向传播和反向传播。
        Args:
            x (numpy.ndarray): 输入数据。
            y (numpy.ndarray): 真实标签。
        Returns:
            float: 损失值。
        """
        # 前向传播
        x = self.linear_layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear_layer2.forward(x)
        x = self.relu2.forward(x)
        loss = self.loss.calculate_loss(x, y)
        # print("loss:{}".format(loss))

        # 反向传播
        loss_grad = self.loss.backward()
        relu2_grad = self.relu2.backward()
        layer2_grad = self.linear_layer2.backward()
        grads = np.multiply(loss_grad, relu2_grad)
        self.linear_layer2.update(grads.T)
        grads = layer2_grad.dot(grads)
        relu1_grad = self.relu1.backward()
        grads = np.multiply(relu1_grad, grads)
        self.linear_layer1.update(grads.T)

        return loss

    def predict(self, x):
        """
        预测函数，返回预测的类别。
        Args:
            x (numpy.ndarray): 输入数据。
        Returns:
            int: 预测的类别。
        """
        # 前向传播
        x = self.linear_layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear_layer2.forward(x)
        x = self.relu2.forward(x)
        number_index = self.loss.predict(x)

        return number_index

    def save(self, path='.', w1_name='w1', w2_name='w2'):
        """
        保存权重参数到文件。
        Args:
            path (str): 保存路径。
            w1_name (str): 第一个权重参数文件名。
            w2_name (str): 第二个权重参数文件名。
        """
        w1 = self.linear_layer1.get_w()
        w2 = self.linear_layer2.get_w()
        np.save(os.path.join(path, w1_name), w1)
        np.save(os.path.join(path, w2_name), w2)

    def evaluate(self, x, y):
        """
        评估函数，判断预测是否正确。
        Args:
            x (numpy.ndarray): 输入数据。
            y (numpy.ndarray): 真实标签。
        Returns:
            bool: 预测是否正确。
        """
        if y == self.predict(x):
            return True
        else:
            return False

    def load_param(self, path=""):
        """
        加载权重参数。
        Args:
            path (str): 参数文件路径。
        """
        w1 = np.load(os.path.join(path, 'w1.npy'))
        w2 = np.load(os.path.join(path, 'w2.npy'))
        self.linear_layer1.set_w(w1)
        self.linear_layer2.set_w(w2)


def one_hot_encoding(y):
    one_hot_y = np.eye(10)[y]

    return one_hot_y


def train_net(data_path=''):
    """
    训练神经网络模型。
    Args:
        data_path (str): 数据路径，默认为当前路径。
    """
    m_net = MNISTNet()

    x_train, y_train = dl.load_train_data(data_path)
    '''
    这行代码执行了以下两个步骤：
    1、归一化：将输入数据中的每个像素值除以255。这是因为MNIST数据集中的像素值范围是0到255，通过将它们除以255，
       将它们缩放到了0到1之间。这种归一化有助于神经网络的训练，因为它将输入数据的值范围映射到了一个更小的区间，有助于加速训练过程。
    2、中心化：从每个像素值中减去0.5。这意味着将所有像素值的平均值平移到0附近。
       这个步骤有助于训练过程中的数值稳定性，通常在输入数据的均值不为0时进行中心化。
    综合起来，这行代码的目的是将原始的像素值转换为一个均值接近0、范围在-0.5到0.5之间的归一化值，
    以更好地满足神经网络的训练需求。这是神经网络中常见的数据预处理步骤之一。
    '''
    x_train = x_train / 255 - 0.5
    '''
    将目标标签（或类别）进行独热编码（One-Hot Encoding）的操作。这是为了适应多类别分类问题的标签表示方式。
    在机器学习中，特别是在多类别分类任务中，通常会使用独热编码来表示目标标签。独热编码是一种二进制编码方式，
    其中每个类别都由一个长度等于类别总数的二进制向量表示。在这个向量中，只有一个元素是1，其他元素都是0，用来表示样本所属的类别。
    这种编码方式有助于模型理解和处理多类别分类问题。
    在这里，one_hot_encoding 函数将原始的目标标签 y_train 转换为独热编码的形式。
    '''
    y_train = one_hot_encoding(y_train)

    epoch = 200  # 定义训练的轮数，也就是训练过程将遍历整个训练数据集的次数
    # num_examples = iter * batch_size
    for i in range(epoch):
        average_loss = 0  # 用于记录每个epoch中的平均损失
        '''
        x_train.shape[0] 是 NumPy 数组 x_train 的属性，表示该数组的第一维度（也就是行数）。
        在这个上下文中，x_train 应该是一个包含训练数据的 NumPy 数组，通常是一个二维数组，其中每一行代表一个训练样本，而每一列代表该样本的特征。
        所以，x_train.shape[0] 返回的值就是训练数据集中样本的数量，也就是数据集的行数。
        在上述代码中，它用于计算平均损失，以便将累积的损失除以数据集中的样本数量，从而得到每个 epoch 的平均损失。
        例如，如果 x_train 的形状是 (60000, 784)，那么 x_train.shape[0] 就是 60000，表示数据集中有 60000 个训练样本。
        '''
        for j in range(x_train.shape[0]):
            # 遍历训练数据集中的每个样本
            # 计算并累加损失，同时更新模型参数
            average_loss += m_net.train(x_train[j], y_train[j])
            # 打印每2000个样本的训练损失，以便实时监视模型的训练进度。
            if j % 2000 == 0:
                print('train set loss(epo:{}): {}'.format(i, average_loss / (j + 1)))
        # 打印每个epoch的平均损失
        print('train set average loss: {}'.format(average_loss / x_train.shape[0]))
        # 保存模型参数，通常在每个epoch结束后保存，以便稍后使用
        m_net.save()


def eval_net(path=""):
    """
    评估神经网络模型。
    计算在测试数据集上的分类准确度（precision）或召回率（recall）
    加载训练好的神经网络模型，在测试数据集上进行分类预测，并计算模型的分类准确度，以评估模型在新数据上的性能。
    Args:
        path (str): 参数文件路径。
    """
    x_test, y_test = dl.load_test_data(path)
    '''
    这行代码执行了以下两个步骤：
    1、归一化：将输入数据中的每个像素值除以255。这是因为MNIST数据集中的像素值范围是0到255，通过将它们除以255，
       将它们缩放到了0到1之间。这种归一化有助于神经网络的训练，因为它将输入数据的值范围映射到了一个更小的区间，有助于加速训练过程。
    2、中心化：从每个像素值中减去0.5。这意味着将所有像素值的平均值平移到0附近。
       这个步骤有助于训练过程中的数值稳定性，通常在输入数据的均值不为0时进行中心化。
    综合起来，这行代码的目的是将原始的像素值转换为一个均值接近0、范围在-0.5到0.5之间的归一化值，
    以更好地满足神经网络的训练需求。这是神经网络中常见的数据预处理步骤之一。
    '''
    x_test = x_test / 255.0 - 0.5
    precision = 0  # 初始化一个变量 precision 为0，用于计算分类准确度。
    m_net = MNISTNet()  # 创建一个新的神经网络模型 m_net，这个模型是用于在测试数据集上进行分类预测的。
    m_net.load_param()  # 加载之前训练好的神经网络模型的参数。这些参数包含了模型在训练数据上学到的权重和偏差。

    for i in range(x_test.shape[0]):  # 遍历测试数据集中的每个样本。
        '''
        对当前测试样本进行分类预测，并与真实标签 y_test[i] 进行比较。
        如果模型的预测结果与真实标签一致，表示分类正确，执行下面的代码块。
        '''
        if m_net.evaluate(x_test[i], y_test[i]):
            precision += 1  # 如果分类正确，将 precision 值加1，用于统计正确分类的样本数量。
    precision /= len(x_test)  # 计算分类准确度。它将正确分类的样本数量除以测试数据集的总样本数量，从而得到分类准确度。
    # 通常，分类准确度是一个在0到1之间的值，表示模型在测试数据上的性能。
    print('precision of test data set is {}'.format(precision))


def visualize(path):
    """
    可视化函数，用于展示模型预测结果。
    Args:
        path (str): 参数文件路径。
    """
    # 加载测试数据
    x, y_gt = dl.load_test_data(path)
    # 数据预处理：将像素值归一化到[-0.5, 0.5]范围
    x_imput = x / 255.0 - 0.5
    # 创建一个新的神经网络模型
    m_net = MNISTNet()
    # 加载之前训练好的模型参数
    m_net.load_param()
    # 随机选择一个测试样本的索引
    visualize_idx = random.randint(0, x.shape[0] - 1)
    # 使用模型对选定的测试样本进行预测
    y_pred = m_net.predict(x_imput[visualize_idx])

    # 创建一个绘图区域，并显示图像和标签信息
    plt.subplot(111)
    title = "真值标签为：{},""预测标签为：{}".format(y_gt[visualize_idx], y_pred)
    plt.title(title, fontproperties='SimHei')
    # 将测试样本的图像转换成可视化的形式，并显示在图中
    img = np.array(x[visualize_idx]).reshape((28, 28))
    # cmap参数: 为调整显示颜色 gray为黑白色，加_r取反为白黑色
    plt.imshow(img, cmap='gray')
    # 显示图像
    plt.show()


"""
这三个函数一起构成了神经网络的训练、评估和可视化过程，能够训练模型、评估其性能并可视化模型的预测结果。
"""
if __name__ == '__main__':
    '''
    train_net() 函数的作用是训练神经网络模型。
    它通过多次迭代训练数据集中的样本，使用反向传播算法来更新神经网络的权重和参数，以最小化损失函数。
    在训练过程中，模型逐渐调整自己以提高对训练数据的拟合。训练完成后，模型的参数将保存在文件中，以备将来使用。
    '''
    # train_net()
    '''
    eval_net() 函数的作用是评估训练好的神经网络模型在测试数据集上的性能。
    它会加载之前训练好的模型参数，并使用这些参数来进行测试数据集上的预测。
    然后，它会计算模型在测试数据集上的精度或其他性能指标，并打印出来，以衡量模型的性能。
    '''
    # eval_net()
    '''
    visualize(args.data_path) 函数的作用是可视化神经网络模型的预测结果。
    它会加载训练好的模型参数，并随机选择一个测试样本，然后使用模型对该样本进行预测。
    最后，它会显示该样本的真实标签和模型的预测标签，以及样本的图像，以帮助可视化模型的性能。
    '''
    visualize(args.data_path)
