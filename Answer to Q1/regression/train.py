from dataset import MNIST, collater
from model import Model
import os
import torch
from torch.utils.data import DataLoader
import numpy

os.environ['CUDA_VISIBLE_DEVICES'] = "0" #specify GPU
reg_model = Model(num_classes=2, num_instances=100).cuda()
reg_model.train()
mnist = MNIST()
mnist_loader = DataLoader(
    mnist,
    batch_size=8,
    collate_fn=collater,
    #   num_workers=8,
    pin_memory=True)

params = [p for p in reg_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=5e-4)

criterion = torch.nn.L1Loss()

num_epoch = 0
max_epoch = 300
iter = 0

for num_epoch in range(max_epoch):
    t_loss = 0
    for data in mnist_loader:
        img, gt = data
        img = img.cuda()
        gt = gt.cuda()

        optimizer.zero_grad()
        y_logits = reg_model(img).view(-1)

        loss = criterion(y_logits, gt)
        loss.backward()
        optimizer.step()

        if numpy.isnan(loss.item()): #Look for the cause in the case of nan
            print(gt)
            print(y_logits)
            exit()

        t_loss += loss.item()
    
    print("Epoch:",num_epoch ,"loss:", t_loss / len(mnist_loader))

torch.save(reg_model,"./final.pth")
