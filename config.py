import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import os
le = LabelEncoder()
root_images="/home/xu/MyData/casia-maxpy-clean/CASIA-maxpy-clean/"
images=[]
labels=[]
dirs=[root_images+d+'/' for d in os.listdir(root_images)]
for d in dirs:
    images.extend([d+imgs for imgs in os.listdir(d)])
for img in images:
    labels.append(img.split('/')[-2])

labels=le.fit_transform(labels)
    
class Config:
    test_list = "/home/xu/MyData/lfw/lfw_test_pair.txt"
    test_root = "/home/xu/MyData/lfw/lfw-align-128"
    device = 'cuda'
    train_batch_size=128
    
    test_batch_size=128
    nums_output=len(dirs)
    images=images
    labels=labels
    dirs=dirs
    train_transform = transforms.Compose([
            
            transforms.ToPILImage(),
           
            transforms.Grayscale(),

            transforms.Resize(128),
           
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.5], std=[0.225])

        ])

    test_transform = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Grayscale(),

        transforms.Resize(128),
      
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.5], std=[0.225])

    ])

    lr=1e-1
    weight_decay=5e-4

