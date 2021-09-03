import torch
from PIL import Image
import os
from natsort import natsorted
from initialize_models import initialize_model
import torchvision.transforms as transforms
import numpy as np
import math

def predict(model,img):

    img = img.unsqueeze(0).cuda()
    out = model(img)
    #prob = torch.softmax(out,dim=1)[0][0].cpu().detach().numpy()
    prob = torch.softmax(out,dim=1)[0].cpu().detach().numpy()
    pred = out.argmax().cpu().numpy()

    return pred,prob,out



if __name__ == "__main__":
    model_name = "resnet101"
    num_classes = 2
    input_size = (100,100)
    
    pretrained = True
    featureExtract = False


    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize(size = (input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



    model = initialize_model(model_name, num_classes, pretrained, input_size,featureExtract)


    model.load_state_dict(torch.load("./checkpoints/model_"+str(model_name) +".pth"))
    model.to(device)
    model.eval()


    test_path = "./data/OK/"
    test_images = natsorted(os.listdir(test_path))

    for idx, image in enumerate(test_images):
        img = Image.open(test_path + image)

        img = data_transforms(img)
        img.to(device)
        with torch.no_grad():
            pred,prob,out = predict(model,img)
        #prob = prob.astype(np.float)
        #prob = np.round(prob,4)

        print(f"Image name: {test_path + image}")
        print(f"Prob: {prob}, Pred: {pred}")
        print("**************************\n")
        
