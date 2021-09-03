from numpy.core.numeric import True_
import torch
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

from initialize_models import initialize_model
from calc_mean_std import calc_mean_std
class AvgMeter():
    def __init__(self):
        self.Losses= []
        self.runningLoss = []
        self.epoch_mean_loss = []

    def reset(self):
        self.epoch_mean_loss = []

def train_model(model_name,model,train_data,input_size,num_epochs,batch_size,lr,num_classes):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    mean,std = calc_mean_std(train_data,input_size,batch_size)

    data_transforms = transforms.Compose([
        transforms.Resize(size = (input_size)),
        #transforms.ColorJitter(brightness=0.6,contrast=0.8,saturation=0.7,hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ])

    trainset = torchvision.datasets.ImageFolder(train_data,transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size,shuffle = True)


    #optimizer = torch.optim.SGD(params_to_update, lr = lr, momentum = momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = 0.05, verbose = True)

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()


    print("[INFO] Training started.")

    mean_losses = []
    #lossMeter = AvgMeter()
    for epoch in range(num_epochs):
        running_loss = []
        #lossMeter.reset()
        loop = tqdm(enumerate(train_loader),total = len(train_loader))
        for batch_idx, (data,target) in loop:
            data, target = data.to(device), target.to(device)

            if model_name == "inception":
                output,aux = model(data)
            else:
                output = model(data)

            optimizer.zero_grad()

            if num_classes == 1:
                loss = criterion(output, target.float())
            else:
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()


            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)
            #lossMeter.runningLoss.append(loss.item())
            #lossMeter.epoch_mean_loss = sum(lossMeter.runningLoss) / len(lossMeter.runningLoss)

            loop.set_description(f"Epoch: [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, 
                                lr = optimizer.param_groups[0]["lr"]) 

        if len(mean_losses) >= 1: 
            if mean_loss < min(mean_losses):
                print("[INFO] Model saved.")
                torch.save(model.state_dict(), model_save_name)
        
        # if len(lossMeter.Losses) >= 1:
        #     if lossMeter.epoch_mean_loss < min(lossMeter.Losses):
        #         print("[INFO] Model saved.")
        #         torch.save(model.state_dict(),model_save_name)


        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)
        #lossMeter.Losses.append(lossMeter.epoch_mean_loss)
        #scheduler.step(lossMeter.epoch_mean_loss)


if __name__ == "__main__":
    
    """ARGS"""
    model_name = "resnet101"
    input_size = (100,100) #height,width
    batch_size = 1
    num_epochs = 6000
    num_classes = 2
    lr = 1e-4
    momentum = 0.9
    
    train_data = "./data/"
    model_save_name = "./checkpoints/model_" + model_name + ".pth"
    
    pretrained = True # pretrained weights that train on Imagenet dataset
    featureExtract = False #When True, Freeze pretrained layers. Just use for feature extract. 
                           #When False, update all params of model.

    
    
    model = initialize_model(model_name, num_classes, pretrained, input_size,featureExtract)
    
    #print(model)
    # counter = 0
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
    #         counter += 1
    # print(counter)
    # params_to_update = model.parameters()
    # #print(params_to_update)
    # if featureExtract:
    #     params_to_update = []
    #     for name,param in model.named_parameters():
    #         #print(f"name: {name}")
    #         if param.requires_grad == True:
    #             params_to_update.append(param)
    
    # else:
    #     for name,param in model.named_parameters():
    #         if param.requires_grad == True:
    #             #print("\t",name)
    #             pass

   

    
    train_model(model_name, 
                model, 
                train_data,
                input_size,
                num_epochs,
                batch_size,
                lr,
                num_classes)


    print("[INFO] Training ended.")
