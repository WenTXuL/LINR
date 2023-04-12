import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from Model.SIREN import *
from util import *

def get_image(fname,noise):    
    #get noise image
    img = (np.array( Image.open(fname)).astype(np.float32)/255-0.5)*2
       
    img= add_gaussian_noise(img, noise*2/255)
    img=np.clip(img,-1,1)  
   
    
    
    # plt.imshow(np.squeeze((img+1)/2))
    
    # plt.pause(0.1)
   
    noisy_tensor = torch.tensor(img).permute(2,0,1)
    return noisy_tensor
#Dataset
class ImageFitting(Dataset):
    def __init__(self, fname,noise):
        super().__init__()
        img = get_image(fname,noise)
        #self.pixels = img.permute(1, 2, 0).view(-1, 1)
        print(img.shape)
       
        self.pixels = img.permute(1, 2, 0).view(-1, 3)
        
        self.coords = get_origin_mgrid(img.shape[1],img.shape[2],dim=2)
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
          
        return self.coords, self.pixels
def train(fname,noise):
  im=Image.open(fname)
  w,h=im.size
  sidelength1=h
  sidelength2=w
  image = ImageFitting(fname,noise)
  dataloader = DataLoader(image, batch_size=1, pin_memory=True, num_workers=0)

  img_siren = Siren(in_features=2, out_features=3, hidden_features=256, 
                  hidden_layers=6, outermost_linear=True)
  img_siren.cuda()
  #total steps
  total_steps = 500
  #test steps
  steps_til_summary = 500
  optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
  model_input, ground_truth = next(iter(dataloader))
  model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
  start_time=time.time()
  for step in range(total_steps+1):
      model_output, coords = img_siren(model_input)    
      loss = ((model_output - ground_truth)**2).mean()
      
      if not step % steps_til_summary:
          print("Step %d, Total loss %0.6f" % (step, loss))
          
          outputimg=model_output.cpu().view(sidelength1,sidelength2,3).detach().numpy()
          gt = (np.array(Image.open(fname).resize([sidelength2,sidelength1])).astype(np.float32)/255)        
          outputimg=(outputimg+1)/2     
          
        
        
          # plt.imshow((model_output.cpu().view(sidelength1,sidelength2,3).detach().numpy()+1)/2)
          # plt.pause(0.1)

          #print metrics
          PSNR = psnr(gt,outputimg)
          print("psnr = ", PSNR)               
          SSIM = ssim(gt,outputimg,multichannel=True )
          print("ssim = ", SSIM)
          end_time=time.time()
          print("time = ",end_time-start_time)
          # save
          savefloder = 'result'
          if not os.path.exists(savefloder): 
            os.makedirs(savefloder)
          
          basename = os.path.basename(fname)
          file_name = os.path.splitext(basename)[0]          
          resultimage=np_to_pil(outputimg.transpose(2,0,1))
          resultimage.save(savefloder+'/'+file_name+'_denoising result.png')
      
      optim.zero_grad()
      loss.backward()
      optim.step()
if __name__ == '__main__':
  fname="Data/Denoising/image_F16_512rgb.png" 
  noise=25 #gaussian noise
  train(fname,noise)