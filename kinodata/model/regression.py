from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from torch_geometric.data import Batch
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
        wandb.init()
        wandb.define_metric("val/pose_mae", summary="min")
        wandb.define_metric("val/activity_mae", summary="min")
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
        return self.model(batch)
   
    def activity_mae(self, pred, batch):
        return (batch.y - pred[:,1]).abs().mean()
    
    def activity_mse(self, pred, batch):
        return (batch.y - pred[:,1]).pow(2).mean()

    def pose_mae(self, pred, batch):
        return (batch.predicted_rmsd - pred[:,0]).abs().mean()

    def pose_mse(self, pred, batch):
        return (batch.predicted_rmsd - pred[:,0]).pow(2).mean()


    def training_step(self, batch, *args) -> Tensor:    #for loop over the bacthes 
        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
       
        # Forward pass for activity batch
        pred_activity = self.forward(activity_batch)
        loss_activity = self.activity_mse(pred_activity, activity_batch)
        self.log("train/loss_activity", loss_activity, batch_size=pred_activity.size(0), on_epoch=True)
    
        # Forward pass for pose batch
        pred_pose = self.forward(pose_batch)
        loss_pose = self.pose_mse(pred_pose, pose_batch)
        self.log("train/loss_pose", loss_pose, batch_size=pred_pose.size(0), on_epoch=True)

        # Combine losses
        total_loss = self.weight_pki * loss_activity + self.weight_pose * loss_pose
        self.log("train/total_loss", total_loss, batch_size=pred_activity.size(0), on_epoch=True)

        wandb.log({
            "train/total_loss": total_loss,
             "batch_size": pred_activity.size(0), #check this is right
             "on_epoch": True
            })
        return total_loss

        
    def validation_step(self, batch, *args, key: str = "val"):
        # Unpack the activity and pose batches directly
        activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader

        # Forward pass for activity batch
        pred_activity = self.forward(activity_batch)
        activity_mae = self.activity_mae(pred_activity, activity_batch)
        self.log(f"{key}/activity_mae", activity_mae, batch_size=pred_activity.size(0), on_epoch=True)
    
        # Forward pass for pose batch
        pred_pose = self.forward(pose_batch)
        target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target
        pose_mae = self.pose_mae(pred_pose, pose_batch)
        self.log(f"{key}/pose_mae", pose_mae, batch_size=pred_pose.size(0), on_epoch=True)

        # Combined MAE of activity and pose
        combined_mae = (activity_mae + pose_mae) / 2
    
        self.log(f"{key}/mae", combined_mae, batch_size=max(pred_activity.size(0), pred_pose.size(0)), on_epoch=True)

        return {
            "pred": torch.cat([pred_activity[:, 1], pred_pose[:, 0]]),  # Concatenate activity and pose predictions
            "target": torch.cat([activity_batch.y, pose_batch.predicted_rmsd]),  # Concatenate activity and pose targets
            f"{key}/mae": combined_mae
            }

    
    
    def process_eval_outputs(self, outputs) -> float:
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
        from torch_geometric.data import Batch

        
        #I think there is an inconsistency here with the batch
        batch=Batch.from_data_list(batch)
        pred = self.forward(batch)
        pred_activity = pred[:, 1]
        pred_unc_activity = pred[:, 0]
        # pose_certainty = pred[:, 2]


        #need to change this to do it properly 28/06
        return {"pred activity": pred_activity, "target": batch.y.flatten()}
    

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
