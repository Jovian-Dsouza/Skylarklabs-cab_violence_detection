import torch
import torchvision
from torchvision.transforms import Resize, RandomCrop, RandomRotation

class ShakyResize(torch.nn.Module):
    def __init__(self, 
                 target_height: int,
                 target_width: int,
                 shakiness: int = 1, 
                 p : float = 0.5):
        
        super().__init__()
        self.p = p
        self.crop = (target_height, target_width)
        self.resize = (target_height+shakiness, target_width+shakiness)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            
            resize_t = Resize(self.resize)
            crop_t = RandomCrop(self.crop)
        
            x = resize_t(x)
            
            aug = []
            for i in range(x.shape[1]):
                aug.append(crop_t(x[:, i, :, :]))
            
            return torch.stack(aug, dim=1)
        
        else:
            resize = Resize(self.crop)
            x = resize(x)
            return x

class ShakyRotate(torch.nn.Module):
    def __init__(self, 
                 degrees : float, 
                 p : float = 0.5,
                 interpolation = torchvision.transforms.InterpolationMode.BILINEAR):
        
        super().__init__()
        self.p = p
        self.degrees = degrees
        self.interpolation = interpolation
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            
            rotate_t = RandomRotation(self.degrees, 
                            interpolation=self.interpolation)
            
            aug = []
            for i in range(x.shape[1]):
                aug.append(rotate_t(x[:, i, :, :]))
            
            return torch.stack(aug, dim=1)
        
        else:
            return x
