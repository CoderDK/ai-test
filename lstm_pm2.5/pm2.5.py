import pandas as pd
from numpy import array
from torch import nn
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

#No：序列号
#year：年份
#month：月份
#day：日
#hour：小时
#pm2.5：PM2.5浓度
#DEWP：露点温度
#TEMP：温度
#PRES：压力
#cbwd：风向
#Iws：风速
#Is：积雪的时间
#Ir：累积的下雨时数

# lstm模型参数
seq_len = 15
input_s = 11
layer_num = 1
hidden_size = 20
learn_rate = 0.00001
epoch_n = 100
epoch_t_n = 5
log_file = "argumentInfo.txt"

# 数据集加载
filepath_train = 'PRSA_data_2010.1.1-2014.12.31_train.csv'
filepath_test = 'PRSA_data_2010.1.1-2014.12.31_test.csv'

data_train = pd.read_csv(filepath_train,index_col=0) 
data_test = pd.read_csv(filepath_test,index_col=0)

# 数据预处理
# 'Is','cbwd','TEMP','year','month','day','hour','Ir','Iws','DEWP','PRES','pm2.5'
dataset_train = data_train.loc[:,['Is','cbwd','TEMP','year','month','day','hour','Ir','Iws','DEWP','PRES','pm2.5']].values / 1000
dataset_test = data_test.loc[:,['Is','cbwd','TEMP','year','month','day','hour','Ir','Iws','DEWP','PRES','pm2.5']].head(50).values / 1000


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return torch.Tensor(array(X)), torch.Tensor(array(y))


# 定义模型
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.input_s = input_s
        self.lstm = nn.LSTM(input_size=self.input_s, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32，对于序列比较短的数据num_layers不要设置大，否则效果会变差
        # 原来的输入格式是：(seq, batch, shape)，设置batch_first=True以后，输入格式就可以改为：(batch, seq, shape)，更符合平常使用的习惯
        self.linear = nn.Linear(self.hidden_size*seq_len, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, self.hidden_size*seq_len)
        x = self.linear(x)
        return x


def lossLine(train_loss_arr,test_loss_arr):
    plt.plot(train_loss_arr, label="loss")
    plt.plot(test_loss_arr, label="test_loss")
    plt.legend(loc='best')
    plt.show()

# 训练模型
def train(model,train_x,train_y,test_x,test_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fun = nn.MSELoss()
    model.train()
    train_loss_arr = []
    test_loss_arr = []
    for epoch in range(epoch_n):
        output = model(train_x)
        loss = loss_fun(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % epoch_t_n == 0 and epoch > 0:
            test_loss = loss_fun(model(test_x), test_y)
            print("epoch:{}, loss:{}, test_loss: {}".format(epoch, loss, test_loss))
            train_loss_arr.append(loss)
            test_loss_arr.append(test_loss)
    return (train_loss_arr,test_loss_arr)

# 测试模型 
def predict(model,test_y,test_x):
    model.eval()
    y = list(test_y*1000)
    y_p = list((model(test_x).data.reshape(-1))*1000)
    
    plt.plot(y, label="real")
    # 原来的走势
    plt.plot(y_p, label="pred")
    # 模型预测的走势
    plt.legend(loc='best')
    plt.show()

    R_2_0 = metrics.r2_score(y, y_p)
    mse = MSE(test_y,model(test_x).data.reshape(-1))
    print("R_2_0: ",R_2_0)
    print("MSE评分：", mse)
    print("MSE评分2：",metrics.mean_squared_error(test_y,model(test_x).data.reshape(-1)))
    return R_2_0,mse

# 评分函数
def MSE(y,y_p):
    return np.sum((np.array(y)-np.array(y_p)) * (np.array(y)-np.array(y_p))) / len(y)


def main():
    # 数据预处理
    train_x,train_y = split_sequences(dataset_train,seq_len)
    test_x,test_y = split_sequences(dataset_test,seq_len)
    # 模型
    model = lstm()
    # 训练并绘制loss图
    train_loss_a,test_loss_a = train(model,train_x,train_y,test_x,test_y)
    lossLine(train_loss_a,test_loss_a)
    # 测试并评估模型
    r_s,mse = predict(model,test_y,test_x)
    info = "input_shpe: " + str(input_s) + \
        " layer: " + str(layer_num) + \
            " hidden: " + str(hidden_size) + " lr: " + str(learn_rate) + \
                " epoch_n: " + str(epoch_n) + " epoch_t_n: " + str(epoch_t_n) + \
                    " r_s: " + str(r_s) + " mse: " + str(mse) + "\n"
    with open(log_file,'a') as f:
        f.write(info)

main()

