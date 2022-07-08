import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
import numpy as np
import timm.optim.optim_factory as optim_factory
import glob

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
        inputs = {"sinograms":sinogram}
        return inputs

class Trainer():
    def __init__(self):
     
        self.loss_avg = 0
        self.log_frequency = 100
        self.device = "cuda"
        self.model = mae_vit_base_patch16_dec512d8b().to(self.device)
        self.dataset = dataset()
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size = 1, shuffle = True,
                num_workers=1, pin_memory=True, drop_last=True, persistent_workers=True)
                
        param_groups = optim_factory.add_weight_decay(self.model, 1e-4)
        self.optimizer = torch.optim.AdamW(param_groups, lr=1e-4, betas=(0.9, 0.95))
        self.scaler = GradScaler()

    
    def log_time(self, batch_idx):

        loss_avg = self.loss_avg/self.log_frequency
        self.loss_avg = 0
        time_sofar = time.time() - self.start_time
        samples_per_sec = (self.step-self.start_step)/time_sofar
        training_time_left = (self.num_total_steps / (self.step-self.start_step) - 1.0) * time_sofar if self.step > 0 else 0
        print_string_time = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string_time.format(self.epoch, batch_idx, samples_per_sec, loss_avg, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def run_one_epoch(self):
    
        for batch_idx, inputs in enumerate(self.dataloader):

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
            
            with autocast():
                print(inputs["sinograms"].shape)
                pred, loss, mask = self.model(inputs["sinograms"])
                print(loss)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()
            #self.model_lr_scheduler.step()

            self.step += 1
            self.loss_avg = self.loss_avg+loss
            
            if self.step % self.log_frequency == 0:
                self.log_time(batch_idx)


if __name__ == '__main__':

    trainer = Trainer()
    trainer.run_one_epoch()