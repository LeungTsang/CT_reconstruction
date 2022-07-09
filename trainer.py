import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
import numpy as np
import timm.optim.optim_factory as optim_factory
import glob
import time
import os
import random

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from model import *

def sec_to_hm_str(t):

    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)


class dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(dataset, self).__init__()
        self.path = sorted(glob.glob("sinodata/*.png"))

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        sinogram = np.array(pil.open(self.path[index]))
        sinogram = torch.Tensor(sinogram)/65535
        theta = torch.linspace(0, np.pi-(np.pi/256), 256)
        
        sinograms = torch.stack([sinogram for _ in range(8)],dim=0)
        thetas = torch.stack([theta+random.uniform(0, np.pi) for _ in range(8)],dim=0)
        inputs = {"sinograms":sinograms, "thetas":thetas}
        return inputs

class Trainer():
    def __init__(self):
     
        self.loss_avg = 0
        self.step = 0
        self.log_frequency = 1000
        self.save_frequency_epoch = 5000
        self.start_time = time.time()
        self.device = "cuda"
        self.save_path = "save_model"

        self.model = mae_vit_base_patch16_dec512d8b().to(self.device)
        self.dataset = dataset()
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size = 1, shuffle = True,
                num_workers=1, pin_memory=True, drop_last=True, persistent_workers=True)
                
        param_groups = optim_factory.add_weight_decay(self.model, 1e-4)
        self.optimizer = torch.optim.AdamW(param_groups, lr=1e-4, betas=(0.9, 0.95))
        self.scaler = GradScaler()

        num_train_samples = len(self.dataloader)
        self.num_total_steps = num_train_samples * 1000
    
    def log_time(self, batch_idx):

        loss_avg = self.loss_avg/self.log_frequency
        self.loss_avg = 0
        time_sofar = time.time() - self.start_time
        samples_per_sec = self.step/time_sofar
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string_time = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string_time.format(self.epoch, batch_idx, samples_per_sec, loss_avg, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.save_path, "weights_{}_{}".format(self.epoch, self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        torch.save(self.model.state_dict(), os.path.join(save_folder, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_folder, "optimizer.pth"))
    
    def run_epoch(self):
    
        for batch_idx, inputs in enumerate(self.dataloader):

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device).squeeze(0)
            
            with autocast():
                pred, loss, mask = self.model(inputs["sinograms"],inputs["thetas"])

                #print(loss)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            #self.model_lr_scheduler.step()

            self.step += 1
            self.loss_avg = self.loss_avg+loss
            
            if self.step % self.log_frequency == 0:
                img = pil.fromarray((255*pred.detach()[0][0].cpu().numpy()).astype(np.uint8))
                img.save("img"+str(self.step)+".png")
                self.log_time(batch_idx)

    def train(self):
        for self.epoch in range(0, 20000):

            self.run_epoch()

            self.epoch = self.epoch+1
            if self.epoch % self.save_frequency_epoch == 0:
                self.save_model()
        
        self.save_model()
                
        return self.model

if __name__ == '__main__':

    trainer = Trainer()
    trainer.train()