import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import h5py as h5
from tqdm import tqdm


#搭建MLP回归模型
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        #定义第一个隐藏层
        self.hidden1=nn.Linear(in_features=1024*1024,out_features=128,bias=True)#8*100 8个属性特征
        #定义第二个隐藏层
        self.hidden2=nn.Linear(128,32)
        
        #回归预测层
        self.predict=nn.Linear(32,18)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        output=self.predict(x)
        return output
mlpreg=MLPregression()
print(mlpreg)

#定义优化器
optimizer=torch.optim.SGD(mlpreg.parameters(),lr=0.01)
loss_func=nn.MSELoss()
train_loss_all=[]
for epoch in tqdm(range(20)):
    train_loss=0
    train_num=0
    for i in tqdm(range(10)):
        with h5.File("train_data/data2.h5", "r") as ipt:
            X_train = np.array(ipt["X"][i*1000:(i+1)*1000], dtype="uint8")
            y_train = np.array(ipt["Y"][i*1000:(i+1)*1000])
            # print(len(ipt["X"]))
        scaler = StandardScaler()
        X_train_s = np.zeros((len(X_train), 1024, 1024))
        for i in range(len(X_train)):
            X_train_s[i] = scaler.fit_transform(X_train[i])
        X_train_s = X_train_s.reshape((-1, 1024*1024))

        #将数据集转化为张量 并处理为PyTorch网络使用的数据
        train_xt=torch.from_numpy(X_train_s.astype(np.float32))
        train_yt=torch.from_numpy(y_train.astype(np.float32))

        #将数据处理为数据加载器
        train_data=Data.TensorDataset(train_xt,train_yt)
        train_loader=Data.DataLoader(dataset=train_data,batch_size=10,shuffle=True,num_workers=0)


        for step,(b_x,b_y) in enumerate(train_loader):
            output=mlpreg(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*b_x.size(0)
            train_num+=b_x.size(0)
            # print(b_x[0].mean(), b_x[0].std())
        train_loss_all.append(train_loss/train_num)
        # print(train_loss/train_num)

# 测试集
with h5.File("train_data/final1.h5", "r") as test:
    X_test = test["X"][:]

#标准化处理
scaler = StandardScaler()
X_test_s = scaler.fit_transform(X_test)
X_test_s = X_test_s.reshape((1, 1024*1024))
test_xt=torch.from_numpy(X_test_s.astype(np.float32))

#预测
pre_y = mlpreg(test_xt)
print(test_xt.mean(), test_xt.std())
pre_y=pre_y.data.numpy()
print(pre_y)
with h5.File("Answer.h5", "w") as opt:
    d = [('SphereId', '<u1'), ('R', '<f8'), ('beta', '<f8',(8,))]
    ans = opt.create_dataset('Answer', (2,), dtype=d)
    ans['SphereId']=np.array([0,1])
    ans['R']=np.hstack((pre_y[0][0],pre_y[0][9]))
    ans[0, 'beta']=pre_y[0][1:9]
    ans[1, 'beta']=pre_y[0][10:18]

'''
pre_y=mlpreg(test_xt)
pre_y=pre_y.data.numpy()
mae=mean_absolute_error(y_test,pre_y)

index=np.argsort(y_test)
plt.figure(figsize=(12,5))
plt.plot(np.arange(len(y_test)),y_test[index],"r",label="original y")
plt.scatter(np.arange(len(pre_y)),pre_y[index],s=3,c="b",label="prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("index")
plt.ylabel("y")
plt.show()
'''
