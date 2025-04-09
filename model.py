import torch

class Autoencoder(torch.nn.Module):
    #意为_ADClust_Autoencoder类的父类是torch.nn.Module

    def __init__(self, input_dim: int, embedding_size: int, act_fn=torch.nn.LeakyReLU ):
        super(Autoencoder, self).__init__() #子类继承了父类的所有属性和方法
        # 构建LeakyReLU函数，函数公式见收藏
        #定义类中的属性

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            act_fn(inplace=True),
            torch.nn.Linear(512, 256),
            act_fn(inplace=True),
            torch.nn.Linear(256, 128),
            act_fn(inplace=True),
            torch.nn.Linear(128, embedding_size))

        #torch.nn.Linear.weight


        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            act_fn(inplace=True),
            torch.nn.Linear(128, 256),
            act_fn(inplace=True),
            torch.nn.Linear(256, 512),
            act_fn(inplace=True),
            torch.nn.Linear(512, input_dim),
        )

    #定义类中的函数
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # ：参数的类型建议符号     -> 返回值的类型建议符号
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
            for batch, _ in trainloader:
                batch_data = batch.to(device)
                reconstruction = self.forward(batch_data)
                loss = loss_fn(reconstruction, batch_data)#损失函数
                optimizer.zero_grad()#将梯度置零
                loss.backward()#反向传播，计算各个参数的梯度
                optimizer.step()#根据上一步的梯度更新自编码器的网络参数，此处梯度越小损失函数值越小，因此通过使梯度减小得到新的网络参数.
