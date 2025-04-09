import torch
import torch.nn as nn
from layer import ZINBLoss, MeanAct, DispAct
import torch.nn.functional as F
from torch.autograd import Variable

class scClust_Autoencoder(torch.nn.Module):

    def __init__(self, input_dim: int, embedding_size: int, act_fn=torch.nn.LeakyReLU,alpha=1.):
        super(scClust_Autoencoder, self).__init__() #子类继承了父类的所有属性和方法
        # 构建LeakyReLU函数，函数公式见收藏
        #定义类中的属性

        self.enc_1 = torch.nn.Linear(input_dim, 512,bias=True)
        self.enc_2 = torch.nn.Linear(512, 256, bias=True)
        self.enc_3 = torch.nn.Linear(256, 128, bias=True)
        self.enc_4 = torch.nn.Linear(128, embedding_size, bias=True)

        self.dec_1 = torch.nn.Linear(embedding_size, 128, bias=True)
        self.dec_2 = torch.nn.Linear(128, 256, bias=True)
        self.dec_3 = torch.nn.Linear(256, 512, bias=True)
        self.dec_4 = torch.nn.Linear(512, input_dim, bias=True)


        self._dec_mean = nn.Sequential(torch.nn.Linear(512, input_dim), MeanAct())
        self._dec_disp = nn.Sequential(torch.nn.Linear(512, input_dim), DispAct())
        self._dec_pi = nn.Sequential(torch.nn.Linear(512, input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss()
        self.alpha=alpha


        #self.encoder = torch.nn.Sequential(
            #torch.nn.Linear(input_dim, 512),
            #act_fn(inplace=True),
            #torch.nn.Linear(512, 256),
            #act_fn(inplace=True),
            #torch.nn.Linear(256, 128),
            #act_fn(inplace=True),
            #torch.nn.Linear(128, embedding_size))

        #self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(embedding_size, 128),
            #act_fn(inplace=True),
            #torch.nn.Linear(128, 256),
            #act_fn(inplace=True),
            #torch.nn.Linear(256, 512),
            #act_fn(inplace=True),
            #torch.nn.Linear(512, input_dim),
        #)

    #定义类中的函数
    def encode(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        embedded = F.relu(self.enc_4(enc_h3))
        # enc_h3 = F.relu(self.enc_3(enc_h2))
        return embedded


    def decode(self, embedded):
        dec_h1 = F.relu(self.dec_1(embedded))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        reconstruction = F.relu(self.dec_4(dec_h3))

        # dec_h3 = F.relu(self.dec_3(dec_h2))

        # x_bar = self.x_bar_layer(dec_h2)
        return reconstruction


    def forward(self, x):
        z = self.encode(x)
        embedded = self.encode(x)
        dec_h1 = F.relu(self.dec_1(embedded))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        mean = self._dec_mean(dec_h3)
        disp = self._dec_disp(dec_h3)
        pi = self._dec_pi(dec_h3)

        return z, mean, disp, pi

    def target_distribution(self,q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()






    #def encode(self, x: torch.Tensor) -> torch.Tensor:
        # ：参数的类型建议符号     -> 返回值的类型建议符号
        #return self.encoder(x)

    #def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        #return self.decoder(embedded)

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
        #z = self.encode(x)
        #print(z.shape)
        #embedded = self.encode(x)
        #reconstruction = self.decode(embedded)
        #reconstruction=int(reconstruction)
        #x=int(x)
        #mean = nn.Sequential(nn.Linear(reconstruction, x), MeanAct())
        #disp = nn.Sequential(nn.Linear(reconstruction, x), DispAct())
        #pi = nn.Sequential(nn.Linear(reconstruction, x), nn.Sigmoid())
        #return z, mean, disp, pi

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(trainloader):
            #for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(trainloader):
                #for batch, _ in trainloader:
                x_tensor = x_batch.to(device)
                x_raw_tensor = x_raw_batch.to(device)
                sf_tensor = sf_batch.to(device)
                z, mean, disp, pi = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
                optimizer.zero_grad()#将梯度置零
                loss.backward()#反向传播，计算各个参数的梯度
                optimizer.step()#根据上一步的梯度更新自编码器的网络参数，此处梯度越小损失函数值越小，因此通过使梯度减小得到新的网络参数.