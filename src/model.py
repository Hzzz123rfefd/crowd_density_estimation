import torch
import torch.nn as nn
import math
from tqdm import tqdm
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.utils import *

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, device = "cuda",bn=False):
        super(MCNN, self).__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn)).to(self.device)
        
        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn)).to(self.device)
        
        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn)).to(self.device)
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn)).to(self.device)
        
    def forward(self, input):
        output = {}
        image = input["image"].to(self.device)
        x1 = self.branch1(image)
        x2 = self.branch2(image)
        x3 = self.branch3(image)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        output["predict"] = x
        output["label"] = input["label"].to(self.device)
        return output
    
    def load_pretrained(self, save_model_dir, lamda = None):
        lamda = self.lamda if lamda == None else lamda
        self.load_state_dict(torch.load(save_model_dir + "/" + str(lamda) + "/model.pth"))
    
    def save_pretrained(self,  save_model_dir, lamda = None):
        lamda = self.lamda if lamda == None else lamda
        torch.save(self.state_dict(), save_model_dir + "/" + str(lamda) + "/model.pth")

    def trainning(
            self,
            train_dataloader:DataLoader = None,
            test_dataloader:DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 1e-4,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            save_checkpoint_step:str = 10,
            save_model_dir:str = "models"
        ):
            ## 1 trainning log path 
            first_trainning = True
            check_point_path = save_model_dir   + "/checkpoint.pth"
            log_path = save_model_dir  + "/train.log"

            ## 2 get net pretrain parameters if need 
            """
                If there is  training history record, load pretrain parameters
            """
            if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
                self.load_pretrained(save_model_dir,self.finetune_model_lamda)  
                first_trainning = False

            else:
                if not os.path.isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                with open(log_path, "w") as file:
                    pass


            ##  3 get optimizer
            if optimizer_name == "Adam":
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            elif optimizer_name == "AdamW":
                optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, 
                mode = "min", 
                factor = factor, 
                patience = patience
            )

            ## init trainng log
            if first_trainning:
                best_loss = float("inf")
                last_epoch = 0
            else:
                checkpoint = torch.load(check_point_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_loss = checkpoint["loss"]
                last_epoch = checkpoint["epoch"] + 1

            try:
                for epoch in range(last_epoch,total_epoch):
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = train_loss + test_loss
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": None,
                            "lr_scheduler": None
                        },
                        check_point_path
                    )

                    if epoch % save_checkpoint_step == 0:
                        os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                        torch.save(
                            {
                                "epoch": epoch,
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict()
                            },
                            save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                        )
                    if is_best:
                        self.save_pretrained(save_model_dir)

            # interrupt trianning
            except KeyboardInterrupt:
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        check_point_path
                    )

    def compute_loss(self, input):
        output = {}
        mse_loss = nn.MSELoss()

        """ reconstruction loss """
        output["total_loss"] = mse_loss(input["predict"], input["label"])
        return output

    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, input in enumerate(test_dataloader):
                """ forward """
                output = self.forward(input)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())

            str = "Test Epoch: {:d}, total_loss: {:.4f}".format(
                epoch,
                total_loss.avg
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg