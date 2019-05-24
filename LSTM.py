import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import numpy as np

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TFF_CPP_MIN_LEVEL'] = '2' # 表示tf的显示等级，第一级显示所有关于tf的信息，第二级只显示warning和error，第三级只显示error。
assert tf.__version__.startswith('2.')

# 数据预处理
def datapreprocess(x, y):
    max_review_len = 80
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
    return x, y

# 参数设置
batchsz = 128
total_words = 10000
hidden_num = 100
max_review_len = 80 # 函数内部也需要修改

# 导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 设置data_batch
x_train, y_train = datapreprocess(x_train, y_train)
x_test, y_test = datapreprocess(x_test, y_test)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(batchsz, drop_remainder=True) # 表示最后一批数据小于batchsz时是否丢掉

# 设定网络结构
class MyLSTM(keras.Model):
    def __init__(self, units):
        super(MyLSTM, self).__init__()
        # 下面写具体的网络层
        # Embedding层用于model的第一层
        # [b, 80] => [b, 80, 100]
        self.embedding=layers.Embedding(total_words, hidden_num, input_length=max_review_len)
        # LSTMCell层,设置两个LSTM层
        # [b, 80, 100] => [b, 64] Sequence => Vector
        self.LSTM0=layers.LSTMCell(units, dropout=0.2)
        self.LSTM1 = layers.LSTMCell(units, dropout=0.2)
        # Dense层，为了表示0～1转换为一个数
        # [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

        self.state0 = [tf.zeros([batchsz, units]),tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units]),tf.zeros([batchsz, units])]
    def call(self, inputs, training=True):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        # 对于Cell类型的要做时间循环
        for word in tf.unstack(x, axis=1): # 表示将每一个word输入到LSTM当中
            out0, state0 = self.LSTM0(word, state0, training)
            out1, state1 = self.LSTM1(out0, state1, training)
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob
def main():
    units = 64
    epochs = 10
    model = MyLSTM(units)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(db_train, epochs=epochs, validation_data=db_test)

if __name__=='__main__':
    main()
