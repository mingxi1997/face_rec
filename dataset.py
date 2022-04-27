from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import Config
import cv2



class MYDataset(Dataset):
    def __init__(self):
       
        self.images=Config.images
        self.labels=Config.labels
      
        
    def __len__(self):
     
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img_root=self.images[idx]
        label=self.labels[idx]
        img=cv2.imread(img_root)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=Config.train_transform(img)

        return img,label
