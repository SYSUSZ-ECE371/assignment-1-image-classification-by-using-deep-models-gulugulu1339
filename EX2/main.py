import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR,LambdaLR,ExponentialLR
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

#冻结模型中的某些层
def freeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if name in layer_names:
            param.requires_grad = False  # 冻结该层
        else:
            param.requires_grad = True   # 其他层正常训练

if __name__ == '__main__':
    # Set data directory
    data_dir = 'data/flower_dataset'

    # Data augmentation and normalization for training and validation
    data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        ### START SOLUTION HERE ###
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomRotation(30),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),  # 随机擦除
        # Add five data augmentation methods, Normalizating and Tranform to tensor
        ### END SOLUTION HERE ###
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Automatically split into 80% train and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use DataLoader for both train and validation datasets
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Get class names from the dataset
    class_names = full_dataset.classes

    # Load pre-trained model and modify the last layer
    model = models.resnet101(pretrained=True)
    freeze_layers(model, ['layer1', 'layer2'])

    # GRADED FUNCTION: Modify the last fully connected layer of model
    ### START SOLUTION HERE ###
    # Modify the last fully connected layer of model
    num_in_features = model.fc.in_features#ResNet 的最后一层是全连接层，名称为 fc
    # model.fc = nn.Linear(num_in_features, len(class_names))
    model.fc = nn.Sequential(
        nn.Linear(num_in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),#防止过拟合
        nn.Linear(512, len(class_names))#输出层输出维度修改为5
    )
    ### END SOLUTION HERE ###


    # GRADED FUNCTION: Define the loss function
    ### START SOLUTION HERE ###
    # Define the loss function
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Define the optimizer
    ### START SOLUTION HERE ###
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ### END SOLUTION HERE ###

    # Learning rate scheduler
    warmup_epochs = 20
    num_epochs = 100
    warmup_lr_init = 0.01
    final_lr = 0.001
    gamma=0.95

    linearLR = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_lr_init/final_lr + (1 - warmup_lr_init/final_lr) * (epoch / warmup_epochs)
    )


    def custom_lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 前十轮线性衰减
            return warmup_lr_init / final_lr + (1 - warmup_lr_init / final_lr) * (epoch / warmup_epochs)
        else:
            # 后面十轮指数衰减
            return (final_lr / warmup_lr_init) * (gamma ** (epoch - warmup_epochs))

    mix_LR= LambdaLR(optimizer,
                     lr_lambda=custom_lr_lambda)

    cosine_annealing = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=2e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[mix_LR, cosine_annealing],
        milestones=[warmup_epochs]#切换策略轮次
    )

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=100,gradient_accumulation_steps=4):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        update_num=1

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Print learning rate for current epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                #会把显存跑满，直接减小batch
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':

                            # GRADED FUNCTION: Backward pass and optimization
                            ### START SOLUTION HERE ###
                            # Backward pass and optimization
                            loss.backward()  # 反向传播
                            # optimizer.step()  # 参数更新
                            ### END SOLUTION HERE ###

                            # 每当累积的梯度达到设定的步数时，更新模型参数
                            if (inputs.size(0) % gradient_accumulation_steps == 0) or (
                                    inputs.size(0) == dataset_sizes[phase] % batch_size):
                                # print(f"更新{update_num}批")
                                update_num+=1
                                optimizer.step()  # 更新参数

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()  # Update learning rate based on scheduler

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save the model if validation accuracy is the best so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model
                    save_dir = 'work_dir/excise2'
                    os.makedirs(save_dir, exist_ok=True)

                    # GRADED FUNCTION: Save the best model
                    ### START SOLUTION HERE ###
                    # Save the best model
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))  # 保存最佳模型权重
                    ### END SOLUTION HERE ###

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model


    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)