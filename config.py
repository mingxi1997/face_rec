import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import os
le = LabelEncoder()
root_images="/home/xu/casia-maxpy-clean/CASIA-maxpy-clean/"
images=[]
labels=[]
dirs=[root_images+d+'/' for d in os.listdir(root_images)]
for d in dirs:
    images.extend([d+imgs for imgs in os.listdir(d)])
for img in images:
    labels.append(img.split('/')[-2])

labels=le.fit_transform(labels)
    
class Config:
    embedding_size=512
    test_list = "/home/xu/XU/lfw/lfw_test_pair.txt"
    test_root = "/home/xu/XU/lfw/lfw-align-128"
    load_net=False
    device = 'cuda'
    arc_s=30
    arc_m=0.5
    base_lr=0.1
    num_epochs=50
    train_batch_size=512
    test_batch_size=512
    multi_gpu=True
    num_iter_show=300
    nums_output=len(dirs)
    images=images
    labels=labels
    dirs=dirs
    test_transform = transforms.Compose([
        
        transforms.ToPILImage(),
        # color_aug,
        transforms.Grayscale(),

        transforms.Resize(128),
     
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.5], std=[0.5])

    ])
    
    train_data_transform = transforms.Compose([
        
        transforms.ToPILImage(),
      
        transforms.Grayscale(),
        transforms.Resize((144, 144)),
        transforms.RandomCrop(128),
     
        transforms.RandomHorizontalFlip(),
     
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])

    ])

    # lr=1e-2
    weight_decay=5e-4
    
    
    
    
    
    
    
    
    
    
    
    
    
    

