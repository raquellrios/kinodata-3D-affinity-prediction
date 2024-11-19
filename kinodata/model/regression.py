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


#class UnicertaintyAwareLoss(nn.Module):
class RegressionModel(pl.LightningModule):


    log_scatter_plot: bool = False
    log_test_predictions: bool = False


    def __init__(self, config: Config, weight_pki=1, weight_pose=1):
        super(RegressionModel, self).__init__() #do I need this?
        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()
	    #self.set_criterion()
	    #self.loss_pki=loss_pki #do I need this?
	    #self.loss_pose=loss_pose #do I need this?
        self.weight_pki = weight_pki
        self.weight_pose = weight_pose
        #wandb.watch(self, log="all", log_freq=10)

    def define_metrics(self):
        #wandb.init(project="kinodata_extended", config=self.config)
        #print('rpintint setting from insde regression')
        #print(wandb.run.settings)
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
                #"monitor": "train/loss_activity",
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
        prob_pose = 1 / (1 + torch.exp(2 * (pose_rmsd - 2.1)))

        return prob_pose
    
    
    def activity_uncertainty_transform(self, activity_unc_raw):

        eps = 1e-8  # Small epsilon to avoid exact 0 or 1

        act_unc = torch.clamp(torch.sigmoid(activity_unc_raw), min=eps, max=1 - eps)

        return act_unc

    
    def compute_loss_activity(self, pred, batch, mask):

        target_activity = batch.y 

        pred_activity = pred[:, 0] 
        pred_unc_activity_raw = pred[:, 1] 
        #pose_certainty = pred[:, 2]


        #I have made this change now because I think that it should be the RMSD of data, not the predicted one but CHECK! okt 15
        pose_rmsd=batch.predicted_rmsd

        pose_certainty = self.rmsd_to_prob_transform(pose_rmsd)
        
        pred_unc_activity = self.activity_uncertainty_transform(pred_unc_activity_raw)


        epsilon = 1e-8
        regulariser_term = 1 / (pred_unc_activity + epsilon)
        
         
        loss_activity = (((target_activity - pred_activity).pow(2) / (pred_unc_activity.pow(2) + epsilon)) * pose_certainty) + regulariser_term #or + pred_unc_activity#
        #loss_activity = (((target_activity - pred_activity).pow(2) * (1 + pred_unc_activity)) * pose_certainty) + pred_unc_activity #regulariser_term #or + pred_unc_activity#
        

        return torch.sum(loss_activity * mask)/torch.sum(mask)


    def compute_loss_pose(self, pred, batch, mask):
         
        target_exp_rmsd=batch.predicted_rmsd
        
        pose_pred = pred[:, 2] 
        
        
        #converting the input into the sigmoid and clamping values 
        target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
        #pose_certainty = self.rmsd_to_prob_transform(pose_pred)

        loss_pose = torch.nn.functional.binary_cross_entropy_with_logits(pose_pred, target_pose_certainty, reduction="none")

        return torch.sum(loss_pose * mask)/torch.sum(mask)
    
    def on_train_start(self):
        # Save initial weights at the start of training
        os.makedirs("checkpoints_initial_one_forward", exist_ok=True)
        initial_weights_path = "checkpoints_initial_one_forward/initial_weights.pt"
        torch.save(self.state_dict(), initial_weights_path)
        print(f"Initial weights saved to {initial_weights_path}")



    def training_step(self, batch, *args) -> Tensor:    #for loop over the bacthes 


        from torch_geometric.data import Batch

        os.makedirs("checkpoints_one_forward_test", exist_ok=True)

        with torch.autograd.detect_anomaly():

    
            # Unpack the activity and pose batches directly
            activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
            n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs

            #print(f"Batch index: {batch_idx}")
            #print(f"Activity batch size: {activity_batch.num_graphs}")
            #print(f"Pose batch size: {pose_batch.num_graphs}")

            total_batch = Batch.from_data_list([activity_batch, pose_batch])
            # Create Boolean masks for activity and pose entries
            activity_mask = torch.cat([torch.ones(n_act, dtype=torch.bool), torch.zeros(n_pose, dtype=torch.bool)]).to(self.device)
            pose_mask = torch.cat([torch.zeros(n_act, dtype=torch.bool), torch.ones(n_pose, dtype=torch.bool)]).to(self.device)

            #print(activity_mask)
            # Forward pass for activity batch
            pred = self.forward(total_batch)

            act_pred=pred[:, 0]
            pose_pred=pred[:,2]
            unc_pred=pred[:,1]
            unc_pred_transformed=self.activity_uncertainty_transform(unc_pred[activity_mask])
            target_exp_rmsd = pose_batch.predicted_rmsd
            target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
            pred_pose = self.rmsd_to_prob_transform(pose_pred[pose_mask])
        

           
            print('target_activity')
            print(activity_batch.y)
            print('pred activity is')
            print(act_pred[activity_mask])
            print("inspection of unc activity pred")
            print(unc_pred_transformed)
            print("uncertainty untrasnformed")
            print(unc_pred[activity_mask])
            print('rmsd_normal')
            print(target_exp_rmsd)
            print("rmsd prediction striaghout of the model")
            print(pose_pred)
            print('pred_pose is')
            print(pred_pose)
            print('target_rmsd is')
            print(target_rmsd)
        

            loss_activity = self.compute_loss_activity(pred, total_batch, activity_mask)
            self.log("train/loss_activity", loss_activity, batch_size=n_act, on_epoch=True, on_step=True)
    
            # Forward pass for pose batch
            #pred_pose = self.forward(pose_batch) #something I am not sure about this forward prediction is which values is i taking into account, I have some nans
	        #for the activity values for example of the pose batch!
            loss_pose = self.compute_loss_pose(pred, total_batch, pose_mask)
            self.log("train/loss_pose", loss_pose, batch_size=n_pose, on_epoch=True, on_step=True)

            # Combine losses
            total_loss = self.weight_pki * loss_activity + self.weight_pose * loss_pose
            #total_loss_normalised = self.weight_pki * (loss_activity - loss_activity.mean())/torch.std(loss_activity) + self.weight_pose * loss_pose.mean()
            self.log(
		    "train/total_loss", 
		    total_loss, 
		    batch_size= n_act + n_pose,
		    on_epoch=True, 
		    on_step=True)
            #wandb.log({"batch_size_total_wandb": n_act + n_pose, "batch_size_pose_wandb": n_pose, "batch_size_activity_wandb": n_act,
            #       }, commit=True)
            #total=n_act + n_pose
            #self.log("batch_size_total", total, batch_size=total, on_epoch=True, on_step=True)
            #self.log("batch_size_pose",  n_pose, batch_size=n_pose, on_epoch=True, on_step=True)
            #self.log("batch_size_activity", n_act , batch_size= n_act , on_epoch=True, on_step=True)


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

            
            #print("printing model parameter")
            #print(dict(self.named_parameters())['out.3.weight'])

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
                torch.save(checkpoint, f"checkpoints_one_forward_test/model_nan_detected_epoch_{self.current_epoch}.pt")

        
                self.trainer.should_stop = True #stop training if NaNs are detected

            else:

                #saving the model and gradients hen everthing runs fine

                checkpoint = {"model_state_dict": self.state_dict(), "gradients": {}}

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        checkpoint["gradients"][name] = param.grad.clone().cpu()  # Save a copy of gradients to avoid issues


                # Save checkpoint
                #if self.current_epoch % 5 == 0:
                if self.current_epoch % 1 == 0 and self.trainer.is_last_batch:

                    torch.save(checkpoint, f"checkpoints_one_forward_test/model_all_ok_epoch_{self.current_epoch}.pt")

            wandb.log({"batch_size_total_wandb": n_act + n_pose, "batch_size_pose_wandb": n_pose, "batch_size_activity_wandb": n_act,
                   })


        return total_loss

        
    def validation_step(self, batch, *args, key: str = "val"):


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
        unc_pred=pred[:,1]
        unc_pred_transformed=self.activity_uncertainty_transform(unc_pred[activity_mask])
        activity_mae = (act_pred[activity_mask] - activity_batch.y).abs().mean()  # Assuming pred_activity[:, 0] corresponds to pred_activity
        self.log(f"{key}/mae_activity", activity_mae, batch_size=n_act, on_epoch=True)

        print('checking activity validation')
        print('target_activity')
        print(activity_batch.y)
        print('pred activity is')
        print(act_pred[activity_mask])
        print("inspection of unc activity pred")
        print(unc_pred_transformed)
        print("uncertainty untrasnformed")
        print(unc_pred[activity_mask])

        errors = torch.abs(act_pred[activity_mask] - activity_batch.y)
        correlation = torch.corrcoef(torch.stack((errors, unc_pred[activity_mask])))[0, 1]
        print("Correlation between error and predicted uncertainty:", correlation.item())
        corr_to_record = correlation.item()
        self.log("val/error_uncertainty_correlation", corr_to_record, on_epoch=True, on_step=True, batch_size=n_act)
        #wandb.log({"val/error_uncertainty_correlation": corr_to_record})
        

        
    
        
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        pred_pose = self.rmsd_to_prob_transform(pose_pred[pose_mask])
        
        pose_mae = (pred_pose - target_rmsd).abs().mean()  
        self.log(f"{key}/pose_mae", pose_mae, batch_size=n_pose, on_epoch=True)

        print('checking pose validation')
        print('rmsd_normal')
        print(target_exp_rmsd)
        print('pred_pose is')
        print(pred_pose)
        print('target_rmsd is')
        print(target_rmsd)
        

        # Combined MAE of activity and pose
        combined_mae = (activity_mae * n_act + pose_mae * n_pose) / (n_act + n_pose)
    
        #print(f"Logging {key}/mae: {combined_mae}")
        self.log(f"{key}/mae", combined_mae, batch_size=n_act + n_pose, on_epoch=True)


        #checking what I log as outputs
        #print("activity output")
        #print(act_pred[activity_mask])
        #print("pose output")
        #print(pred_pose)


        # Return predictions and targets for evaluation
        #return {
        #    	"pred": torch.cat([act_pred[activity_mask],pred_pose]) ,  # Concatenate activity and pose predictions
		#        "target":  torch.cat([activity_batch.y, target_rmsd]),   # Concatenate activity and pose targets
        #        f"{key}/mae": combined_mae
        #}
    
        return {
            	"pred_activity": act_pred[activity_mask] ,
		        "target_activity": activity_batch.y,   # Concatenate activity and pose targets
                f"{key}/mae": activity_mae, 
                "pred_pose": pred_pose, 
                "target_pose": target_rmsd, 
                f"{key}/mae": pose_mae
             }

    
    
    def process_eval_outputs(self, outputs) -> float:
        
        pred_activity = torch.cat([output["pred_activity"] for output in outputs], 0)
        target_activity = torch.cat([output["target_activity"] for output in outputs], 0)

        pred_pose = torch.cat([output["pred_pose"] for output in outputs], 0)
        target_pose = torch.cat([output["target_pose"] for output in outputs], 0)


        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
            pred_activity.std() * target_activity.std()
        ).cpu().item()

        activity_mae = (pred_activity - target_activity).abs().mean()

        pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
            pred_pose.std() * target_pose.std()
        ).cpu().item()
        pose_mae = (pred_pose - target_pose).abs().mean()

        return pred_activity, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae
    

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred_activity, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs(outputs)
        self.log("val/corr_activity", activity_corr)
        self.log("val/corr_pose", pose_corr)

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

            y_min = min(pred_pose.min().cpu().item(), target_pose.min().cpu().item()) - 1
            y_max = max(pred_pose.max().cpu().item(), target_pose.max().cpu().item()) + 1
            fig, ax = plt.subplots()
            ax.scatter(target_pose.cpu().numpy(), pred_pose.cpu().numpy(), s=0.7)
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("Pred")
            ax.set_xlabel("Target")
            ax.set_title(f"pose corr={pose_corr}")
            wandb.log({"scatter_val_pose": wandb.Image(fig)})
            plt.close(fig)


    def predict_step(self, batch, *args):
        
        print('I am in the predict_step')
        from torch_geometric.data import Batch

        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs
        total_batch = Batch.from_data_list([activity_batch, pose_batch])

        activity_mask = torch.cat([torch.ones(n_act, dtype=torch.bool), torch.zeros(n_pose, dtype=torch.bool)]).to(self.device)
        pose_mask = torch.cat([torch.zeros(n_act, dtype=torch.bool), torch.ones(n_pose, dtype=torch.bool)]).to(self.device)

        pred = self.forward(total_batch)

        act_pred=pred[:, 0]
        unc_pred=pred[:, 1]
        pose_pred=pred[:,2]

        
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
        pred_pose = self.rmsd_to_prob_transform(pose_pred[pose_mask])
        

        return {"pred activity": act_pred[activity_mask], "target": activity_batch.y, "pred unc activity": unc_pred[activity_mask],  
		        "pred pose": pred_pose, "pose_target": target_rmsd
                }
    

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch, key="test")
        return info

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred_activity, pred_pose, target_activity, target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs(outputs)
        self.log("test/mae_activity", activity_mae)
        self.log("test/corr_activity", activity_corr)
        self.log("test/mae_pose", pose_mae)
        self.log("test/corr_pose", pose_corr)

        if self.log_test_predictions:
            test_predictions = wandb.Artifact("test_predictions", type="predictions")
            #data = cat_many(outputs, subset=["pred", "ident"])
            data = cat_many(outputs, subset=["pred_activity", "target_activity", "pred_pose", "target_pose"])
            values = [t.detach().cpu() for t in data.values()]
            values = torch.stack(values, dim=1)
            table = wandb.Table(columns=list(data.keys()), data=values.tolist())
            test_predictions.add(table, "predictions")
            wandb.log_artifact(test_predictions)
            pass
