from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import os

#qqplot
import statsmodels.api as sm


from kinodata.configuration import Config
from kinodata.model.resolve import resolve_loss
from kinodata.model.resolve import resolve_optim


def cat_many(
    data: List[Dict[str, Tensor]], subset: Optional[List[str]] = None, dim: int = 0
) -> Dict[str, Tensor]:
    if subset is None:
        subset = list(data[0].keys())
    assert set(subset).issubset(data[0].keys())

    def ensure_tensor(sub_data, key):
        if isinstance(sub_data[key], torch.Tensor):
            return sub_data[key]
        if isinstance(sub_data[key], list):
            # what have i done
            return torch.tensor([int(x) for x in sub_data[key]])
        raise ValueError(sub_data, key, "cannot convert to tensor")

    return {
        key: torch.cat([ensure_tensor(sub_data, key) for sub_data in data], dim=dim)
        for key in subset
    }



class RegressionModel(pl.LightningModule):


    log_scatter_plot: bool = False
    log_test_predictions: bool = False


    def __init__(self, config: Config, initial_weight_pki=1, initial_weight_pose=0, pose_threshold=0.4): 
        super(RegressionModel, self).__init__() #do I need this?
        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()
        self.use_one_forward = True

        # Initial weights
        self.initial_weight_pose = initial_weight_pose
        self.initial_weight_pki = initial_weight_pki

        # Dynamic weights
        self.current_weight_pose = initial_weight_pose
        self.current_weight_pki = initial_weight_pki

        #pose threshold --> ill use corr
        self.pose_threshold = pose_threshold

        

         
        
    def define_metrics(self):
        wandb.define_metric("val/mae", summary="min")
        wandb.define_metric("val/corr", summary="max")


    def configure_optimizers(self):
        Opt = resolve_optim(self.hparams.optim)
        optim = Opt(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )

        return [optim], [
            {
                "scheduler": scheduler,
                "monitor": "val/mae",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    
    def forward(self, batch) -> Tensor:
        
        pred = self.model(batch)

        return pred
    
    
    def rmsd_to_prob_transform(self, pose_rmsd):
    
        #prob_pose = 1 / (1 + torch.exp(torch.clamp(2 * (pose_rmsd - 2.2 ), min=-50, max=50)))
        #prob_pose = 1 / (1 + torch.exp(torch.clamp(2 * (pose_rmsd - 1.5 ), min=-50, max=50)))
        #prob_pose = 1 / (1 + torch.exp(2 * (pose_rmsd - 2.1)))
        prob_pose = 1 / (1 + torch.exp(0.5 * (pose_rmsd - 4)))

        return prob_pose
    

    def update_weights(self):

        pass

        # Reduce the pose weight and increase activity weight
        

        #if self.current_weight_pose > 1:
        #    self.current_weight_pose = self.current_weight_pose - (self.current_weight_pose * 0.2)  # Reduce by 20% of the current weight
        #    #self.current_weight_pose = self.initial_weight_pose * 0.2 # this would reduce it to a fix number- check what is better if the aboce or this!
        #else:
        #    self.current_weight_pose=self.current_weight_pose#

        #self.current_weight_pki = self.initial_weight_pki + (self.initial_weight_pose - self.current_weight_pose)
        #self.log("weights_adjusted", True)

        
    
    def compute_loss_activity_mask(self, pred, batch, mask):

        target_activity = batch.y 

        pred_activity = pred[:, 0] 

        log_uncertainty_act = pred[:, 1]
        #pose_certainty = pred[:, 2] 
        #variance = torch.exp(log_uncertainty_act) #sigma **2
        variance = torch.clamp(torch.exp(log_uncertainty_act), min=1e-3, max=10) #sigma **2
        
        

        #pose_pred = pred[:, 2]
        #pose_certainty = self.rmsd_to_prob_transform(pose_pred)


        ######to delete
        pose_rmsd=batch.predicted_rmsd

        pose_certainty = self.rmsd_to_prob_transform(pose_rmsd)

        
        epsilon = 1e-8 #for numerical stability

        #lambda_reg = 1e-3  # Small regularisation weight--> adjust accordingly
        #lambda_reg = 2e-3  # Small regularisation weight--> adjust accordingly
        #regularisation_term = 1 / (variance + epsilon)
        #masked_regularisation_term = regularisation_term * mask  
        #regularisation_loss = lambda_reg * (torch.sum(masked_regularisation_term) / torch.sum(mask))  # Masked mean


        nll = 0.5 * torch.log(2 * torch.pi * variance + epsilon) + 0.5 * (torch.pow((target_activity - pred_activity), 2) / (variance + epsilon))
      
        loss_activity = pose_certainty * nll 

        return torch.sum(loss_activity * mask)/torch.sum(mask) #+ regularisation_loss
  

    def compute_loss_activity_nomask(self, pred, batch):

        target_activity = batch.y 

        pred_activity = pred[:, 0] 

        log_uncertainty_act = pred[:, 1]
        

        #variance = torch.clamp(torch.exp(log_uncertainty_act), min=1e-3, max=10) #sigma **2
        variance = torch.exp(log_uncertainty_act)

        
        #pose_pred=batch.predicted_rmsd
        #print("rmsd predicted")  
        #pose_certainty=self.rmsd_to_prob_transform(pose_pred)
        #print(pose_certainty)
        #pose_pred = pred[:, 2] #detach the pose certainty - remobing this gradient here 
        #pose_certainty = self.rmsd_to_prob_transform(pose_pred).detach() 



        
        epsilon = 1e-8 #for numerical stability
        #regularisation_term = 1 / (variance + epsilon)

        #nll = 0.5 * torch.log(2 * torch.pi * variance + epsilon) + 0.5 * (torch.pow((target_activity - pred_activity), 2) / (variance + epsilon))
        #nll =  torch.pow((target_activity - pred_activity), 2) / (variance + epsilon) #+ regularisation_term 
        #nll = 0.5 * torch.log(variance + epsilon) + 0.5 * (torch.pow((target_activity - pred_activity), 2) / (variance + epsilon))
        


        #loss_activity = pose_certainty * nll 
        #loss_activity = nll 
        loss_fn = torch.nn.GaussianNLLLoss()
        loss_activity=loss_fn(pred_activity, target_activity, variance)

        #loss_activity=nn.MSELoss()(target_activity, pred_activity) #mse for overfit

        #return torch.mean(loss_activity)
        return loss_activity


    def compute_loss_pose_mask(self, pred, batch, mask):
         
        target_exp_rmsd=batch.predicted_rmsd
        
        pose_pred = pred[:, 2] 
        
        
        #converting the input into the sigmoid  
        target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
        pred_pose_certainty = self.rmsd_to_prob_transform(pose_pred)

        loss_pose=torch.nn.functional.binary_cross_entropy(pred_pose_certainty, target_pose_certainty, reduction="none")

        return torch.sum(loss_pose * mask)/torch.sum(mask)
    


    def compute_loss_pose_nomask(self, pred, batch):
         
        target_exp_rmsd=batch.predicted_rmsd
        
        pose_pred = pred[:, 2] 

       
        
        
        #converting the input into the sigmoid  
        target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
        pred_pose_certainty = self.rmsd_to_prob_transform(pose_pred)

        print("target rmsd")
        print(target_pose_certainty)
        print("pose pred")
        print(pred_pose_certainty)

        loss_pose=torch.nn.functional.binary_cross_entropy(pred_pose_certainty, target_pose_certainty, reduction="none")
        
        return torch.mean(loss_pose)
    



    def training_step(self, batch, *args) -> Tensor:
     
        if self.use_one_forward:
            
            return self.train_one_forward(batch)
        else:
            
            return self.train_two_forward(batch)



    def train_one_forward(self, batch, *args) -> Tensor:    #for loop over the bacthes 


        from torch_geometric.data import Batch


        with torch.autograd.detect_anomaly():

    
            # Unpack the activity and pose batches directly
            activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
            n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs


            total_batch = Batch.from_data_list([activity_batch, pose_batch])
            

            # Manually combine scalar/sequence attributes
            total_batch.pocket_sequence = activity_batch.pocket_sequence + pose_batch.pocket_sequence
            
		    # Convert list to tensor, then concatenate

            total_batch.smiles = activity_batch.smiles + pose_batch.smiles
            total_batch.activity_type = activity_batch.activity_type + pose_batch.activity_type.tolist()
            total_batch.scaffold = activity_batch.scaffold + pose_batch.scaffold

            # Create Boolean masks for activity and pose entries
            activity_mask = torch.cat([torch.ones(n_act, dtype=torch.bool), torch.zeros(n_pose, dtype=torch.bool)]).to(self.device)
            pose_mask = torch.cat([torch.zeros(n_act, dtype=torch.bool), torch.ones(n_pose, dtype=torch.bool)]).to(self.device)

            
            pred = self.forward(total_batch)
            

            variance=torch.exp(pred[activity_mask][:, 1])
            

            self.log("predicted_variance", torch.mean(variance), batch_size=n_act, on_epoch=True)
           
            

            loss_activity = self.compute_loss_activity_mask(pred, total_batch, activity_mask)
            self.log("train/loss_activity", loss_activity, batch_size=n_act, on_epoch=True, on_step=True)
            
            #self.log("nll_mean_term", torch.mean((target_activity - pred_activity) ** 2 / variance, on_step=True))
            #self.log("nll_variance_term", torch.mean(torch.log(2 * torch.pi * variance)), on_epoch=True)
 
      
            loss_pose = self.compute_loss_pose_mask(pred, total_batch, pose_mask)
            self.log("train/loss_pose", loss_pose, batch_size=n_pose, on_epoch=True, on_step=True)

            # Combine losses
            total_loss = self.initial_weight_pki * loss_activity + self.initial_weight_pose * loss_pose


            #total_loss_normalised = self.weight_pki * (loss_activity - loss_activity.mean())/torch.std(loss_activity) + self.weight_pose * loss_pose.mean()
            self.log(
		    "train/total_loss", 
		    total_loss, 
		    batch_size= n_act + n_pose,
		    on_epoch=True, 
		    on_step=True)

 
            batch_size_test = n_act + n_pose
            if batch_size_test != int(batch_size_test):  # Check if batch size is decimal
                print("Warning: Decimal batch size detected!")
                print("n_act:", n_act, "n_pose:", n_pose, "batch_size:", batch_size_test)

            if n_act != int(n_act):  # Check if batch size is decimal
                print("Warning: Decimal act batch size detected!")
                print("n_act:", n_act, "n_pose:", n_pose, "batch_size:", batch_size_test)

            if n_pose != int(n_pose):  # Check if batch size is decimal
                print("Warning: Decimal n_pose batch size detected!")
                print("n_act:", n_act, "n_pose:", n_pose, "batch_size:", batch_size_test)


            # Check if NaNs are in the loss
            if torch.isnan(total_loss).any() or any(torch.isnan(param.grad).any() for param in self.parameters() if param.grad is not None):
                
                print("NaN detected in loss or gradients, saving checkpoint with gradients.")

                 # Save model parameters and gradients
                checkpoint = {"model_state_dict": self.state_dict(),
                               "gradients": {}}
                
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        checkpoint["gradients"][name] = param.grad.clone().cpu()  # Save a copy of gradients to avoid issues

                # Save checkpoint
                torch.save(checkpoint, f"checkpoints_one_forward_test_ll/model_nan_detected_epoch_{self.current_epoch}.pt")

        
                self.trainer.should_stop = True #stop training if NaNs are detected

            #else:

                #saving the model and gradients hen everthing runs fine

                #checkpoint = {"model_state_dict": self.state_dict(), "gradients": {}}

                #for name, param in self.named_parameters():
                    #if param.grad is not None:
                        #checkpoint["gradients"][name] = param.grad.clone().cpu()  # Save a copy of gradients to avoid issues


                # Save checkpoint
                #if self.current_epoch % 5 == 0:
                #if self.current_epoch % 1 == 0 and self.trainer.is_last_batch:

                 #   torch.save(checkpoint, f"checkpoints_one_forward_test_ll/model_all_ok_epoch_{self.current_epoch}.pt")

            wandb.log({"batch_size_total_wandb": n_act + n_pose, "batch_size_pose_wandb": n_pose, "batch_size_activity_wandb": n_act,
                   })


        return total_loss


    def train_two_forward(self, batch, *args) -> Tensor:    #for loop over the bacthes 


        from torch_geometric.data import Batch

        os.makedirs("checkpoints_two_forward_test", exist_ok=True)

        with torch.autograd.detect_anomaly():

    
            # Unpack the activity and pose batches directly
            activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
            n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs


            pred_act = self.forward(activity_batch)

            variance=torch.clamp(torch.exp(pred_act[:,1]), min=1e-3, max=10) #sigma **2

            

            self.log("predicted_variance", torch.mean(variance), batch_size=n_act, on_epoch=True)


            loss_activity = self.compute_loss_activity_nomask(pred_act, activity_batch)
            self.log("train/loss_activity", loss_activity, batch_size=n_act, on_epoch=True, on_step=True)

            epsilon = 1e-8
            
            self.log("nll_mean_term", torch.mean((activity_batch.y - pred_act[:,0]) ** 2 / variance), batch_size=n_act, on_step=True, on_epoch=True)
            self.log("regulariser", torch.mean(torch.log(variance + epsilon)), batch_size=n_act, on_step=True, on_epoch=True)
            #self.log("nll_variance_term", torch.mean(torch.log(2 * torch.pi * variance)), on_step=True)
            # Forward pass for pose batch
            #pred_pose = self.forward(pose_batch) 
            

            #loss_pose = self.compute_loss_pose_nomask(pred_pose, pose_batch)
            #self.log("train/loss_pose", loss_pose, batch_size=n_pose, on_epoch=True, on_step=True)

            # Combine losses
            total_loss = self.current_weight_pki * loss_activity #+ self.current_weight_pose * loss_pose
            
            self.log(
		    "train/total_loss", 
		    total_loss, 
		    #batch_size= n_act + n_pose,
		    batch_size = n_act,
		    on_epoch=True, 
		    on_step=True)



            #batch_size_test = n_act + n_pose
            #if batch_size_test != int(batch_size_test):  # Check if batch size is decimal
            #    print("Warning: Decimal batch size detected!")
            #    print("n_act:", n_act, "n_pose:", n_pose, "batch_size:", batch_size_test)

            # Check if NaNs are in the loss
            if torch.isnan(total_loss).any() or any(torch.isnan(param.grad).any() for param in self.parameters() if param.grad is not None):
                
                print("NaN detected in loss or gradients, saving checkpoint with gradients.")

                 # Save model parameters and gradients
                checkpoint = {"model_state_dict": self.state_dict(),
                              "gradients": {}}
                
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        checkpoint["gradients"][name] = param.grad.clone().cpu()  # Save a copy of gradients to avoid issues

                 #Save checkpoint
                torch.save(checkpoint, f"checkpoints_two_forward_test/model_nan_detected_epoch_{self.current_epoch}.pt")

        
                self.trainer.should_stop = True #stop training if NaNs are detected

            else:

                #saving the model and gradients hen everthing runs fine

                checkpoint = {"model_state_dict": self.state_dict(), "gradients": {}}

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        checkpoint["gradients"][name] = param.grad.clone().cpu()  # Save a copy of gradients to avoid issues


                # Save checkpoint
               
                if self.current_epoch % 5 == 0 and self.trainer.is_last_batch:

                    torch.save(checkpoint, f"checkpoints_two_forward_test/model_all_ok_epoch_{self.current_epoch}.pt")

            #wandb.log({"batch_size_total_wandb": n_act + n_pose, "batch_size_pose_wandb": n_pose, "batch_size_activity_wandb": n_act,
            #       })
            wandb.log({"batch_size_total_wandb": n_act})
            
            # Log current weights
            self.log("train/weight_pose", self.current_weight_pose, batch_size= n_pose, on_epoch=True, on_step=False)
            self.log("train/weight_pki", self.current_weight_pki, batch_size= n_act, on_epoch=True, on_step=False)



        return total_loss
    

    def validation_step(self, batch, *args, key: str = "val"):
        
        if self.use_one_forward:
            #torch.save(self.state_dict(), "checkpoints_one_forward_test/one_forward_before_validation.pt")
            return self.validate_one_forward(batch, key)
        else:
            #torch.save(self.state_dict(), "checkpoints_two_forward_test/two_forward_before_validation.pt")
            return self.validate_two_forward(batch, key)
        


        
    def validate_one_forward(self, batch, *args, key: str = "val"):


        from torch_geometric.data import Batch

        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs
        total_batch = Batch.from_data_list([activity_batch, pose_batch])

        

        activity_mask = torch.cat([torch.ones(n_act, dtype=torch.bool), torch.zeros(n_pose, dtype=torch.bool)]).to(self.device)
        pose_mask = torch.cat([torch.zeros(n_act, dtype=torch.bool), torch.ones(n_pose, dtype=torch.bool)]).to(self.device)

        pred = self.forward(total_batch)

        act_pred=pred[:, 0]
        pose_pred=pred[:,2]
        pred_log_variance=pred[:,1]
        pred_variance = torch.exp(pred_log_variance)
        
        activity_mae = (act_pred[activity_mask] - activity_batch.y).abs().mean()  # Assuming pred_activity[:, 0] corresponds to pred_activity
        self.log(f"{key}/mae_activity", activity_mae, batch_size=n_act, on_epoch=True)

        
        residuals = torch.abs(act_pred[activity_mask] - activity_batch.y)
        #correlation matrix
        correlation_matrix = torch.corrcoef(torch.stack((residuals, pred_variance[activity_mask])))
        correlation = correlation_matrix[0, 1] 
        self.log("uncertainty_correlation", correlation, batch_size=n_act, on_epoch=True)
        

    
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        pred_pose = self.rmsd_to_prob_transform(pose_pred[pose_mask])

        print("=======================================")

        print("checking validation")
        print("activity target")
        print(activity_batch.y)
        print("predicted activity")
        print(act_pred[activity_mask])
        print('rmsd_normal')
        print(target_exp_rmsd)
        print('pred_pose is')
        print(pred_pose)
        print('target_rmsd is')
        print(target_rmsd)  
        print("predicted_variance")
        print(pred_variance[activity_mask])


        
        pose_mae = (pred_pose - target_rmsd).abs().mean()  
        self.log(f"{key}/pose_mae", pose_mae, batch_size=n_pose, on_epoch=True)
        print("printing val values")
        print("pose_mae")
        print(pose_mae)

        print("=======================================")

        
        
        

        # Combined MAE of activity and pose
        combined_mae = (activity_mae * n_act + pose_mae * n_pose) / (n_act + n_pose)
        #combined_mae=activity_mae
      
    
        self.log(f"{key}/mae", combined_mae, batch_size=n_act + n_pose, on_epoch=True)

    
        return {
            	"pred_activity": act_pred[activity_mask] ,
		        "target_activity": activity_batch.y,
                "variance" :  pred_variance[activity_mask],
                "residuals" : residuals,  
                f"{key}/act_mae": activity_mae, 
                "pred_pose": pred_pose, 
                "target_pose": target_rmsd, 
                f"{key}/pose_mae": pose_mae
             }
    

    def validate_two_forward(self, batch, *args, key: str = "val"):


        from torch_geometric.data import Batch

        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs
        
       

        # Forward pass for activity batch
        pred_activity = self.forward(activity_batch)
        act_pred = pred_activity[:,0]
        pred_log_variance=pred_activity[:,1]
        pred_variance = torch.exp(pred_log_variance)
        
        activity_mae = (act_pred - activity_batch.y).abs().mean()  # Assuming pred_activity[:, 0] corresponds to pred_activity
        self.log(f"{key}/mae_activity", activity_mae, batch_size=n_act, on_epoch=True)

        residuals = torch.abs(act_pred - activity_batch.y)
        #correlation matrix
        correlation_matrix = torch.corrcoef(torch.stack((residuals, pred_variance)))
        correlation = correlation_matrix[0, 1] 
        self.log("uncertainty_correlation", correlation, batch_size=n_act, on_epoch=True)
        
        
        
        # Forward pass for pose batch
        pred_pose_raw = self.forward(pose_batch)
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        pred_pose = self.rmsd_to_prob_transform(pred_pose_raw[:, 2])
        
        #pose_mae = (pred_pose - target_rmsd).abs().mean()  
        #self.log(f"{key}/pose_mae", pose_mae, batch_size=n_pose, on_epoch=True)

        print("=======================================")

        print("checking validation")
        print("activity target")
        print(activity_batch.y)
        print("predicted activity")
        print(act_pred)
        print('rmsd_normal')
        print(target_exp_rmsd)
        print('pred_pose is')
        print(pred_pose)
        print('target_rmsd is')
        print(target_rmsd)  
        print("predicted_variance")
        print(pred_variance)


        # Combined MAE of activity and pose
        #combined_mae = (activity_mae * n_act + pose_mae * n_pose) / (n_act + n_pose)
        combined_mae=activity_mae
        self.log(f"{key}/mae", combined_mae, batch_size=n_act + n_pose, on_epoch=True)
       
    
        return {
            	"pred_activity": pred_activity[:, 0],
		        "target_activity": activity_batch.y,   # Concatenate activity and pose targets
                "variance": pred_variance,
                "residuals": residuals,
                f"{key}/act_mae": activity_mae, 
                #"pred_pose": pred_pose, 
                #"target_pose": target_rmsd, 
                #f"{key}/pose_mae": pose_mae
             }

    

    
    
    def process_eval_outputs_one_forward(self, outputs) -> float:
        
        pred_activity = torch.cat([output["pred_activity"] for output in outputs], 0)
        target_activity = torch.cat([output["target_activity"] for output in outputs], 0)

        #pred_pose = torch.cat([output["pred_pose"] for output in outputs], 0)
        #target_pose = torch.cat([output["target_pose"] for output in outputs], 0)
        residuals = torch.cat([output["residuals"] for output in outputs], 0)
        variance = torch.cat([output["variance"] for output in outputs], 0)


        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
            pred_activity.std() * target_activity.std()
        ).cpu().item()

        activity_mae = (pred_activity - target_activity).abs().mean()

        #pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
        #    pred_pose.std() * target_pose.std()
        #).cpu().item()
        #pose_mae = (pred_pose - target_pose).abs().mean()

        print("investigating correlation pose")
        print("target")
        #print(target_pose)
        print("pose pred")
        #print(pred_pose)
        print("correlation pose")
        #print(pose_corr)

        #return pred_activity, variance, residuals, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae
        return pred_activity, variance, residuals,  target_activity,  activity_corr,  activity_mae
    


    def process_eval_outputs_two_forward(self, outputs) -> float:
        
        pred_activity = torch.cat([output["pred_activity"] for output in outputs], 0)
        target_activity = torch.cat([output["target_activity"] for output in outputs], 0)

        #pred_pose = torch.cat([output["pred_pose"] for output in outputs], 0)
        #target_pose = torch.cat([output["target_pose"] for output in outputs], 0)
        residuals = torch.cat([output["residuals"] for output in outputs], 0)
        variance = torch.cat([output["variance"] for output in outputs], 0)


        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
            pred_activity.std() * target_activity.std()
        ).cpu().item()

        activity_mae = (pred_activity - target_activity).abs().mean()

        #pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
        #    pred_pose.std() * target_pose.std()
        #).cpu().item()
        #pose_mae = (pred_pose - target_pose).abs().mean()

        print("investigating correlation pose")
        print("target")
        #print(target_pose)
        print("pose pred")
        #print(pred_pose)
        print("correlation pose")
        #print(pose_corr)

        #return pred_activity, variance, residuals, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae
        return pred_activity, variance, residuals,  target_activity, activity_corr, activity_mae
    

    def validation_epoch_end(self, outputs):
        if self.use_one_forward:
            # Process outputs for one-forward
            pred_activity, variance, residuals, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs_one_forward(outputs)
        else:
            # Process outputs for two-forward
            #pred_activity, variance, residuals, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs_two_forward(outputs)
            pred_activity, variance, residuals,  target_activity, activity_corr,  activity_mae = self.process_eval_outputs_two_forward(outputs)
    
        # Common logging
        self.log("val/corr_activity", activity_corr)
        #self.log("val/corr_pose", pose_corr)


        #if pose_corr >= self.pose_threshold:
        #    self.update_weights()
            
        #checking residuals
        #fig, ax = plt.subplots()
        #ax.scatter(residuals.cpu().numpy() , variance.cpu().numpy() , alpha=0.5)
        #ax.set_xlabel("Absolute Residuals")
        #ax.set_ylabel("Predicted Variance")
        #ax.set_title("Residuals vs. Predicted Variance")
        #wandb.log({"Residuals vs Variance": wandb.Image(plt)})
        #plt.close(fig)

        #Q-QPlot

        if activity_corr > 0.9:

            target_activity = target_activity.detach().cpu().numpy() if isinstance(target_activity, torch.Tensor) else target_activity
            pred_activity = pred_activity.detach().cpu().numpy() if isinstance(pred_activity, torch.Tensor) else pred_activity
            variance = variance.detach().cpu().numpy() if isinstance(variance, torch.Tensor) else variance

            # Compute residuals
            residuals = target_activity - pred_activity

            # Compute predicted standard deviations
            y_std = np.sqrt(variance)

            # Standardize residuals
            standardized_residuals = residuals / y_std

            # Create a Q-Q plot against the standard normal distribution
            fig = sm.qqplot(standardized_residuals, line='45', fit=True)
            plt.title("Q-Q Plot of Standardized Residuals")
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Empirical Quantiles")
            wandb.log({"Q-Q Plot of Standardized Residuals": wandb.Image(plt)})
            plt.close(fig)


        if self.log_scatter_plot:
            y_min = min(pred_activity.min().cpu().item(), target_activity.min().cpu().item()) - 1
            y_max = max(pred_activity.max().cpu().item(), target_activity.max().cpu().item()) + 1
            fig, ax = plt.subplots()
            ax.scatter(target_activity.cpu().numpy(), pred_activity.cpu().numpy(), s=0.7)
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("Pred")
            ax.set_xlabel("Target")
            ax.set_title(f"activity corr={activity_corr}")
            wandb.log({"scatter_val_activity": wandb.Image(fig)})
            plt.close(fig)

            #y_min = min(pred_pose.min().cpu().item(), target_pose.min().cpu().item()) - 1
            #y_max = max(pred_pose.max().cpu().item(), target_pose.max().cpu().item()) + 1
            #fig, ax = plt.subplots()
            #ax.scatter(target_pose.cpu().numpy(), pred_pose.cpu().numpy(), s=0.7)
            #ax.set_xlim(y_min, y_max)
            #ax.set_ylim(y_min, y_max)
            #ax.set_ylabel("Pred")
            #ax.set_xlabel("Target")
            #ax.set_title(f"pose corr={pose_corr}")
            #wandb.log({"scatter_val_pose": wandb.Image(fig)})
            #plt.close(fig)


    def predict_step(self, batch, *args):
        if self.use_one_forward:
            return self.predict_one_forward(batch)
        else:
            return self.predict_two_forward(batch)



    def predict_one_forward(self, batch, *args):
        
        
        from torch_geometric.data import Batch

        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs
        total_batch = Batch.from_data_list([activity_batch, pose_batch])

        activity_mask = torch.cat([torch.ones(n_act, dtype=torch.bool), torch.zeros(n_pose, dtype=torch.bool)]).to(self.device)
        pose_mask = torch.cat([torch.zeros(n_act, dtype=torch.bool), torch.ones(n_pose, dtype=torch.bool)]).to(self.device)

        pred = self.forward(total_batch)

        act_pred=pred[:, 0]
        unc_pred=torch.exp(pred[:, 1])
        pose_pred=pred[:,2]

        
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        pred_pose = self.rmsd_to_prob_transform(pose_pred[pose_mask])
        

        return {"pred activity": act_pred[activity_mask], "target": activity_batch.y, "pred unc activity": unc_pred[activity_mask],  
		        "pred pose": pred_pose, "pose_target": target_rmsd
                }
    
    def predict_two_forward(self, batch, *args):
        
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs

         
        pred_act = self.forward(activity_batch)


        act_pred=pred_act[:, 0]
        log_unc_pred=pred_act[:, 1]
        variance=torch.exp(log_unc_pred)
        


        #pred_pose_raw = self.forward(pose_batch)

        #target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        #target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        #pred_pose = self.rmsd_to_prob_transform(pred_pose_raw[:, 2])
        


        return {"pred activity": act_pred, "target": activity_batch.y, "pred unc activity": variance,  
		        #"pred pose": pred_pose, "pose_target": target_rmsd
                }
    

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch, key="test")
        return info
    

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        #pred_activity, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs(outputs)
        pred_activity, target_activity,  activity_corr,  activity_mae = self.process_eval_outputs(outputs)
    
        self.log("test/mae_activity", activity_mae)
        self.log("test/corr_activity", activity_corr)
        #self.log("test/mae_pose", pose_mae)
        #self.log("test/corr_pose", pose_corr)

        if self.log_test_predictions:
            test_predictions = wandb.Artifact("test_predictions", type="predictions")
            #data = cat_many(outputs, subset=["pred_activity", "target_activity", "pred_pose", "target_pose"])
            data = cat_many(outputs, subset=["pred_activity", "target_activity"])
            values = [t.detach().cpu() for t in data.values()]
            values = torch.stack(values, dim=1)
            table = wandb.Table(columns=list(data.keys()), data=values.tolist())
            test_predictions.add(table, "predictions")
            wandb.log_artifact(test_predictions)
            pass
