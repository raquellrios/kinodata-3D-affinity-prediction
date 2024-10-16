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

   
    
    def compute_loss_activity(self, pred, batch):

        target_activity = batch.y #double check this is right but I think so 
        
        
       
        pred_activity = pred[:, 0]
        pred_unc_activity = pred[:, 1]
        #pose_certainty = pred[:, 2]


        #I have made this change now because I think that it should be the RMSD of data, not the predicted one but CHECK! okt 15
        pose_rmsd=batch.predicted_rmsd 
        pose_certainty = 1 / (1 + torch.exp(torch.clamp(5 * (pose_rmsd - 3), min=-50, max=50))) #What about using the posit probability instead so thatit is actually a pose certainty, since this poses are generated in silico



        epsilon = 1e-8
        regulariser_term = 1 / (pred_unc_activity + epsilon)
        
        #loss_activity = (target_activity - pred_activity).pow(2) / (pred_unc_activity.pow(2) + epsilon)))
	loss_activity = (target_activity - pred_activity).pow(2)
        #loss_activity = (((target_activity - pred_activity).pow(2) / (pred_unc_activity.pow(2) + epsilon)) * pose_certainty) + regulariser_term

        return loss_activity


    def compute_loss_pose(self, pred, batch):
         
        target_exp_rmsd=batch.predicted_rmsd 
        
        pose_certainty = pred[:, 2]
        
        #converting the input into the sigmoid and clamping values
        # I can get rid of the line below when using the cross entropy logit func 
        target_pose_certainty = 1 / (1 + torch.exp(torch.clamp(5 * (target_exp_rmsd - 3), min=-50, max=50))) #changing this to 3 since most values are actually quite tiny. need to think a bit about this further
        
        #target_pose_certainty = 1 / (1 + torch.exp(5 * (target_exp_rmsd - 3))) 
        #target_pose_certainty=1 / (1 + torch.exp(10 * (target_exp_rmsd - 2.5)))

        loss_pose = torch.nn.functional.binary_cross_entropy_with_logits(pose_certainty, target_pose_certainty)
	    #loss_pose = -(target_pose_certainty * torch.log(pose_certainty) + (1 - target_pose_certainty) * torch.log(1 - pose_certainty))

        return loss_pose



    def training_step(self, batch, *args) -> Tensor:    #for loop over the bacthes 

        #from torch_geometric.data import Batch


        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs
	
       
        #I want to talk about why these two predictions are different!
        # Forward pass for activity batch
        pred_activity = self.forward(activity_batch)

        loss_activity = self.compute_loss_activity(pred_activity, activity_batch)
        self.log("train/loss_activity", loss_activity.mean(), batch_size=n_act, on_epoch=True, on_step=True)
    
        # Forward pass for pose batch
        pred_pose = self.forward(pose_batch) #something I am not sure about this forward prediction is which values is i taking into account, I have some nans
	#for the activity values for example of the pose batch!
        loss_pose = self.compute_loss_pose(pred_pose, pose_batch)
        self.log("train/loss_pose", loss_pose.mean(), batch_size=n_pose, on_epoch=True, on_step=True)

        # Combine losses
        total_loss = self.weight_pki * loss_activity.mean() + self.weight_pose * loss_pose.mean()
        self.log(
		"train/total_loss", 
		total_loss, 
		batch_size=n_act + n_pose,
		on_epoch=True, 
		on_step=True)
        wandb.log({"batch_size_activity": n_act, 
                   "batch_size_pose": n_pose}, commit=False)

        return total_loss

        
    def validation_step(self, batch, *args, key: str = "val"):


        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader

        # Forward pass for activity batch
        pred_activity = self.forward(activity_batch)
        activity_mae = (pred_activity[:, 0] - activity_batch.y).abs().mean()  # Assuming pred_activity[:, 0] corresponds to pred_activity
        self.log(f"{key}/activity_mae", activity_mae, batch_size=pred_activity.size(0), on_epoch=True)
    
        # Forward pass for pose batch
        pred_pose = self.forward(pose_batch)
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

        target_rmsd = 1 / (1 + torch.exp(torch.clamp(5 * (target_exp_rmsd - 3), min=-50, max=50))) # Transform target_exp_rmsd to target_rmsd
        pose_mae = (pred_pose[:, 2] - target_rmsd).abs().mean()  
        self.log(f"{key}/pose_mae", pose_mae, batch_size=pred_pose.size(0), on_epoch=True)

        # Combined MAE of activity and pose
        combined_mae = (activity_mae + pose_mae) / 2 #do I need to normalise by anything? check simplified code
    
        print(f"Logging {key}/mae: {combined_mae}")
        self.log(f"{key}/mae", combined_mae, batch_size=max(pred_activity.size(0), pred_pose.size(0)), on_epoch=True)


        # Return predictions and targets for evaluation
        return {
            	"pred": torch.cat([pred_activity[:, 0] , pred_pose[:, 2]]),  # Concatenate activity and pose predictions
		        "target": torch.cat([activity_batch.y, target_rmsd]),  # Concatenate activity and pose targets
                f"{key}/mae": combined_mae
        }

    
    
    def process_eval_outputs(self, outputs) -> float:
        print('printing outputs')
        print(outputs)
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae
    

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("val/corr", corr)

        if self.log_scatter_plot:
            y_min = min(pred.min().cpu().item(), target.min().cpu().item()) - 1
            y_max = max(pred.max().cpu().item(), target.max().cpu().item()) + 1
            fig, ax = plt.subplots()
            ax.scatter(target.cpu().numpy(), pred.cpu().numpy(), s=0.7)
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("Pred")
            ax.set_xlabel("Target")
            ax.set_title(f"corr={corr}")
            wandb.log({"scatter_val": wandb.Image(fig)})
            plt.close(fig)


    def predict_step(self, batch, *args):
        #pred = self.forward(batch,3).flatten()
        print('I am in the predict_step')
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
        n_act, n_pose = activity_batch.size(0), pose_batch.size(0)

        

        pred_activity_i = self.forward(activity_batch)
        pred_activity = pred_activity_i[:, 0]
        pred_unc_activity = pred_activity_i[:,1]
        pred_pose_i = self.forward(pose_batch)
        pred_pose=pred_pose_i[:, 2]

	    #I need to think about this prediction step, regarding the batch do I need to use the same batch for this prediction?


        return {"pred activity": pred_activity, "target": activity_batch.y.flatten(), "pred unc activity": pred_unc_activity,  
		        "pred pose": pred_pose, "pose_target": pose_batch.predicted_rmsd
        }
    

    def test_step(self, batch, *args, **kwargs):
        print('i am in the test_step')
        info = self.validation_step(batch, key="test")
        return info

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("test/mae", mae)
        self.log("test/corr", corr)

        if self.log_test_predictions:
            test_predictions = wandb.Artifact("test_predictions", type="predictions")
            data = cat_many(outputs, subset=["pred", "ident"])
            values = [t.detach().cpu() for t in data.values()]
            values = torch.stack(values, dim=1)
            table = wandb.Table(columns=list(data.keys()), data=values.tolist())
            test_predictions.add(table, "predictions")
            wandb.log_artifact(test_predictions)
            pass
