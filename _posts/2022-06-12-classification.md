---
layout: post
title: "[DL] Classification"
subtitle: Project : Cafe returning robot
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [Deep Learning, pytoch]
comments: true

---



## Cafe returning robot

- 카페에서 음료를 제조해주는 것에서 더 나아가 다 먹고 반납한 트레이를 정리해주는 로봇

`바리스타봇 시연영상`

![img](https://lh3.googleusercontent.com/svea_nGOQg0jXp6MKNV7UZzsUkxpeT6nNGFQPi0UPipvZ6GEDBfwpOwa8BOZTPytnkMy1UrgJ6Ez9gcQFqcN_I9qmirkko4ofxW1n-jaycCzEN8w8bWgKnyjJ6yv4KHhQD3Z_GOBadbgYuPkqGbRIg)


{: .box-note} 
여러 object를 알맞은 장소로 정리하기 위해서는 먼저 object의 class를 분류해야 한다. 



## Class

- cup
- cutlery
- food
- dish
- trash



## Train code

### `import`

```python
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import random
import os, glob
import numpy as np
import time, copy
from PIL import Image
```



### `preprocessing`

```python
# Import dataset
data_dir = '/path/'
print('Folders :', os.listdir(data_dir))
# print("****************************************")
classes = os.listdir(data_dir + "/train")
print('classes :', classes)


train_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = ImageFolder(data_dir + '/train', transform = train_transform)
print('Size of training dataset :', len(dataset))
test_ini = os.path.join(data_dir, 'val')
   
test_set = ImageFolder(data_dir + '/val', transform = test_transform)
print('Size of test dataset :', len(test_set))
train_set = dataset  
print(len(train_set), len(test_set))  

batch_size = 4
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)
```

input 이미지는 64 * 64 사이즈로 작게 resize 해주었고, 데이터의 수가 적어 발생하는 overfitting을 피하기 위해서 `transforms.RandomHorizontalFlip()` 를 추가해주었다.



### `VGG16`

```python
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.model = models.vgg16(pretrained=True)
        # print(self.model)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1024)
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 4)
        )
        
    def forward(self,x):
            # self.param()
            out = self.model(x)
            out = self.classifier(out)
            final = F.softmax(out)
            return final
```

백본 네트워크로는 가장 기본적인 바닐라 모델 VGG16을 사용하였고, 사전에 학습된 pretrained 모델을 활용하였다. (정확도가 약 20% 증가)



### `Train`

```python
lr = 0.001
criterion = nn.CrossEntropyLoss()
model = VGG()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

class TrainModel():
    def __init__(self,model, criterion, optimizer, trainloader, valloader, num_epochs=10):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)
        self.trainloader =trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.best_acc_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc =0.0

        print('## Start learning!! ##')
        for epoch in range(1, self.num_epochs+1):
            
            epoch_loss, epoch_acc = self.train()
            if epoch % 10 ==0 :
                print('Epoch {}/{}'.format(epoch, self.num_epochs))
                print('train | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            epoch_loss, epoch_acc = self.val()
            if epoch % 10 ==0 :
                print('val | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                
		# 가장 높은 정확도를 가진 모델 저장
        model.load_state_dict(self.best_acc_wts)
        
    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # print(targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.cpu().numpy()
            pred = outputs.max(1, keepdim=True)[1]
            # print(pred, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()

        epoch_loss = train_loss /len(self.trainloader.dataset)
        epoch_acc = correct / len(self.trainloader.dataset)
        return epoch_loss, epoch_acc 
    
    def val(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        # total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.valloader):
                # transforms.ToTensor()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                val_loss += nn.functional.cross_entropy(outputs, targets,reduction='sum').item()
                
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
                      
            epoch_loss = val_loss /len(self.valloader.dataset)
            epoch_acc = correct / len(self.valloader.dataset)
           
            
            if epoch_acc >= self.best_acc:
                self.best_acc = epoch_acc
                self.best_acc_wts = copy.deepcopy(self.model.state_dict())

            return epoch_loss, epoch_acc     

       
TrainModel(model, criterion=criterion, optimizer=optimizer,trainloader=train_loader,valloader=test_loader,num_epochs=50)    
```

총 50번의 epoch을 통해서 학습을 진행하였고 정확도는 약 80~90% 정도로 나왔다.

학습을 진행하면서 가장 높은 accuracy를 가진 모델을 최종적으로 저장해준다. 



### `model save`

```python
PATH = './weights/'

torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
```

`.pt` 형태로 저장할 시 모델 내부 파라미터와 모델 전체가 한 번에 저장된다. 

{: .box-note} 
테스트 파일을 따로 만들 시 필요한 과정



### `Test`

```python
test_test_set = ImageFolder(data_dir + '/test', transform = test_transform)
test_test_loader = DataLoader(test_test_set, 1, num_workers=2, pin_memory=True)

def test(model,testloader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    # total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            outputs = model(inputs)
            pred = outputs.argmax(1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            
        # epoch_loss = test_loss /len(testloader.dataset)
        epoch_acc = correct / len(testloader.dataset)
        print('test | Acc: {:.4f}'.format(epoch_acc))

test(model,test_test_loader,criterion)       
```

마지막으로 train과 val에서 사용하지 않았던 새로운 데이터에 대해 테스트를 진행해준다. 

