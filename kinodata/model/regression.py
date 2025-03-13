from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import os
import pandas as pd


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


    def __init__(self, config: Config, initial_weight_pki=1, initial_weight_pose=1, pose_threshold=0.4): 
        super(RegressionModel, self).__init__() #do I need this?
        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()

        self.training_step_outputs = {"activity": [], "pose": []}
        self.test_step_outputs = {"activity": [], "pose": []}

        #self.use_one_forward = True

        # Initial weights
        self.initial_weight_pose = initial_weight_pose
        self.initial_weight_pki = initial_weight_pki

        # Dynamic weights
        self.current_weight_pose = initial_weight_pose
        self.current_weight_pki = initial_weight_pki

        #pose threshold --> ill use corr
        self.pose_threshold = pose_threshold

        # Accumulators for losses per dataset
        self.loss_activity_running = []
        self.loss_pose_running = []

        # Making trainable the rmsd transformation
        #self.rmsd_scale = nn.Parameter(torch.tensor(0.5)) #making it learnable
        #self.rmsd_shift = nn.Parameter(torch.tensor(4.5))


	##directory output name
        self.directory_csv_name = "soft/normalised_act" #"learnable_rmsd/shift_set/clamp"

        

         
        
    def define_metrics(self):
        wandb.define_metric("val/mae_activity", summary="min")
        wandb.define_metric("val/corr_activity", summary="max")


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
                "monitor": "val/mae_activity",
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
        #prob_pose = 1 / (1 + torch.exp(0.5 * (pose_rmsd - 4)))


        #new tries
        #prob_pose = 1 / (1 + torch.exp( 1.5 * (pose_rmsd - 4.5))) #steep
        prob_pose = 1 / (1 + torch.exp( 0.7 * (pose_rmsd - 4.5)))#soft
        #prob_pose = 1 / (1 + torch.exp( 1 * (pose_rmsd - 6))) #m. soft
        #prob_pose = 1 / (1 + torch.exp( 0.7 * (pose_rmsd - 6))) # v. soft

        #prob_pose = 1 / (1 + torch.exp( self.rmsd_scale * (pose_rmsd - self.rmsd_shift)))
        #prob_pose = 1 / (1 + torch.exp( self.rmsd_scale * (pose_rmsd - 4.5))) 

        
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


    def log_metrics_per_rmsd_interval(self, pred_probs, target_probs, bs, key_prefix="val"):

        # Define RMSD intervals
        intervals = [(0.0, 0.2), (0.2, 0.8), (0.8, 1.0)]
   
        #intervals = [(0.0, 0.2),  (0.8, 1.0)]
        for low, high in intervals:
            # Mask for the current interval
            mask = (target_probs >= low) & (target_probs < high)
        
            # Skip if no data points in this range
            if mask.sum()> 0:

                # Extract data for this interval
                pred_interval = pred_probs[mask]
                target_interval = target_probs[mask]

                # Compute metrics
                mae = (pred_interval - target_interval).abs().mean()
                #corr = torch.corrcoef(torch.stack([pred_interval, target_interval]))[0, 1]
                corr = ((pred_interval - pred_interval.mean()) * (target_interval - target_interval.mean())).mean() / (
                pred_interval.std() * target_interval.std()
                ).cpu().item()

                # Log metrics
                self.log(f"{key_prefix}/mae_rmsd_[{low}_{high}]", mae, batch_size=bs, on_epoch=True)
                self.log(f"{key_prefix}/corr_rmsd_[{low}_{high}]", corr, batch_size=bs, on_epoch=True)


    def evaluate_per_scaffold(self, predictions, targets, scaffolds):
        """
        Evaluate pose prediction performance per scaffold.
       
        """
        # Convert to numpy for easier manipulation
        predictions = predictions.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        # Combine into a DataFrame
        data = pd.DataFrame({
            "scaffold": scaffolds,
            "prediction": predictions,
            "target": targets
        })

        # Group by scaffold
        scaffold_groups = data.groupby("scaffold")

        # Calculate metrics for each scaffold
        results = []
        for scaffold, group in scaffold_groups:

            
            mae = (group["prediction"] - group["target"]).abs().mean()  # Mean Absolute Error
            pose_corr = ((group["prediction"] - group["prediction"].mean()) * (group["target"] - group["target"].mean())).mean() / (
            group["prediction"].std() * group["target"].std())  # Pearson Correlation
            count = len(group)  # Number of samples for this scaffold
            results.append({"scaffold": scaffold, "mae": mae, "correlation": pose_corr, "count": count})

        # Convert to DataFramescaffolds
        results_df = pd.DataFrame(results).sort_values(by="count", ascending=False)

        return results_df
    

    def plot_unc_act_pose(self, pose, act_unc, epoch, name):

        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))

        contour = sns.kdeplot(
                x=pose,
                y=act_unc,
                cmap="Blues",
                fill=True,
                levels=20,
                ax=ax,

             )

        # Fix colorbar issue
        mappable = contour.collections[-1]  # Get last contour for colorbar

        fig.colorbar(mappable, ax=ax, label="Density")  # Attach the colorbar

        plt.xlabel("Pose Quality")
        plt.ylabel("Predicted Uncertainty (σ)")
        plt.xlim(0, 1)
        plt.title(f"Contour Plot of Uncertainty vs. Pose Quality at {epoch}")
        wandb.log({f" Act unc and pose calibration contour plot {name}": wandb.Image(plt)})
        plt.close(fig)



    def calibration_plot(self, std_variance, residuals, name):

        #percentiles = np.linspace(0, 100, 10)
        #bin_edges = np.percentile(std_variance, percentiles)  # Find bin edges
        bin_edges = np.linspace(std_variance.min(), std_variance.max(), 11)
        bin_indices = np.digitize(std_variance, bin_edges, right=True)  # Assign σ values to bins
            
             
        mean_pred_sigma = []
        mean_abs_error = []
        #residuals=residuals.cpu().numpy()


            
        for i in range(1, 11):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_pred_sigma.append(np.mean(std_variance[mask]))
                mean_abs_error.append(np.mean(residuals[mask]))



        plt.figure(figsize=(6, 6))
        plt.plot(mean_pred_sigma, mean_abs_error, 'bo-', label="Model Calibration")
        plt.plot([0, max(mean_pred_sigma)], [0, max(mean_pred_sigma)], 'r--', label="Perfect Calibration (y=x)")
        plt.xlabel("Predicted Uncertainty (σ)")
        plt.ylabel("Actual Absolute Error |y - ŷ|")
        plt.title("Uncertainty Calibration Plot")
        plt.legend()
        plt.grid()
        # Log to Weights & Biases (W&B)
        wandb.log({f"Calibration plot {name}": wandb.Image(plt)})
        plt.close()


    def qqplot(self, residuals, y_std, name):
        
        standardized_residuals = (residuals / y_std)

      
        # Generate Q-Q plot
        fig = sm.qqplot(standardized_residuals, line='45', fit=True)
        plt.title(f"Q–Q Plot of Training Residuals (Epoch {self.current_epoch})")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Empirical Quantiles")
            
        # Log to Weights & Biases (W&B)
        wandb.log({f"Q-Q Plot of Standardized Residuals {name}": wandb.Image(plt)})
        plt.close(fig)

    def pose_distribution(self, pose, name):

        plt.hist(pose, bins=20)
        plt.xlabel("Pose quality prediction")
        plt.xlim(0, 1)
        plt.ylabel("Count")
        wandb.log({f"Predicted pose distribution {name}": wandb.Image(plt)})
        plt.close()

    def act_distribution(self, act, name):

        plt.hist(act, bins=50)
        plt.xlabel("Activity prediction")
        plt.ylabel("Count")
        wandb.log({f"Predicted Activity distribution {name}": wandb.Image(plt)})
        plt.close()

  

    def compute_loss_activity_nomask(self, pred, batch):

        target_activity = batch.y 

        pred_activity = pred[:, 0] 

        log_uncertainty_act = pred[:, 1]

        pose_pred = pred[:, 2]
        
        

        #gaussian torch loss

        variance = torch.exp(log_uncertainty_act)


        loss_fn = torch.nn.GaussianNLLLoss(reduction="none")
        loss_activity_nopose=loss_fn(pred_activity, target_activity, variance)

        pose_certainty=self.rmsd_to_prob_transform(pose_pred).detach()

        loss_activity = pose_certainty*loss_activity_nopose

        return torch.mean(loss_activity)

        # non torch loss

        # variance = torch.clamp(torch.exp(log_uncertainty_act), min=1e-3, max=10) #sigma **2
        
        
        
        #pose_pred = pred[:, 2] #detach the pose certainty - remobing this gradient here 
        #pose_certainty = self.rmsd_to_prob_transform(pose_pred).detach() 

        

        
        # epsilon = 1e-8 #for numerical stability
        # regularisation_term = 1 / (variance + epsilon)

        # nll = 0.5 * torch.log(2 * torch.pi * variance + epsilon) + 0.5 * (torch.pow((target_activity - pred_activity), 2) / (variance + epsilon))+ regularisation_term 
        # #nll =  torch.pow((target_activity - pred_activity), 2) / (variance + epsilon) + regularisation_term 
        
        # loss_activity = pose_certainty * nll 

        # #return torch.mean(loss_activity)

        # return loss_activity.mean()

        ##self.log("nll_mean_term", torch.mean((activity_batch.y - pred_act[:,0]) ** 2 / variance), batch_size=n_act, on_step=True, on_epoch=True)
        #    #self.log("regulariser", torch.mean(1 / variance), batch_size=n_act, on_step=True, on_epoch=True)
        #    #self.log("nll_variance_term", torch.mean(torch.log(2 * torch.pi * variance)), on_step=True)


        #kinodata of loss

        #loss_activity=nn.MSELoss()(target_activity, pred_activity) 

        #return loss_activity




    def compute_loss_pose_nomask(self, pred, batch):
         
        target_exp_rmsd=batch.predicted_rmsd
        
       
        #converting the input into the sigmoid  
        #target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
    
        #weights = torch.where(
        #(target_pose_certainty <= 0.1) | (target_pose_certainty >= 0.9),  # extreme cases
        # 5.0,  # Higher weight
        # 1.0   # Default weight for non-extreme cases
        #)
        
        
        pred_pose_logit=pred[:, 2]


        #converting the input into the sigmoid  
        target_pose_certainty = self.rmsd_to_prob_transform(target_exp_rmsd)
        pred_pose=self.rmsd_to_prob_transform(pred_pose_logit) 

        #loss_pose=torch.nn.functional.binary_cross_entropy_with_logits(pred_pose_logit, target_pose_certainty)#, weight=weights)
        loss_pose=torch.nn.functional.binary_cross_entropy(pred_pose, target_pose_certainty)#, weight=weights)
        #loss = nn.MSELoss()

        #loss_pose=loss(pred_pose, target_pose_certainty)
        
        return loss_pose

    def on_training_epoch_start(self):
        """Reset training storage at the start of each epoch."""
        self.training_step_outputs = {"activity": [], "pose": []}

    def training_step(self, batch, batch_idx, dataloader_idx=0, *args) -> Tensor:

        """
        Alternating optimization: Updates one dataset per step.

        """

        #print(f"Batch index: {batch_idx}, Dataloader index: {dataloader_idx}, Batch type: {type(batch)}")
        #print(batch)  

        if "pose" not in self.training_step_outputs:
            self.training_step_outputs["pose"] = []

        if "activity" not in self.training_step_outputs:
            self.training_step_outputs["activity"] = []


        if batch["activity"]:  # Dataset 1 (Activity)

            
            batch_activity= batch["activity"]
           

            pred_act = self.forward(batch_activity)

            ##variance=torch.clamp(torch.exp(pred_act[:,1]), min=1e-3, max=10) #sigma **2
            variance = torch.exp(pred_act[:,1]).detach()

            

            self.log("predicted_variance", torch.mean(variance), batch_size=batch_activity.num_graphs, on_epoch=True)


            loss_activity = self.compute_loss_activity_nomask(pred_act, batch_activity)
            self.log("train/loss_activity", loss_activity, batch_size=batch_activity.num_graphs, on_epoch=True, on_step=True)

            self.log("batch_act", batch_activity.num_graphs, batch_size=batch_activity.num_graphs, on_epoch=False, on_step=True)

            mae_training= (pred_act[:,0] - batch_activity.y).abs().mean() 
            self.log("act_mae_training", mae_training, batch_size=batch_activity.num_graphs, on_epoch=True, on_step=True)


            act_corr_training = ((pred_act[:,0] - pred_act[:,0].mean()) * (batch_activity.y - batch_activity.y.mean())).mean() / (
            pred_act[:,0].std() * batch_activity.y.std())
            self.log("act_corr_training", act_corr_training, batch_size=batch_activity.num_graphs, on_epoch=True, on_step=True)
            self.log("train/weight_pki", self.current_weight_pki, batch_size= batch_activity.num_graphs, on_epoch=True, on_step=False)

            

            ##training q-q plot data

            

            self.training_step_outputs["activity"].append({
                    "pred_activity": pred_act[:,0].detach(),
                    "target_activity": batch_activity.y.detach(),
                    "variance": variance,
                    "pose":self.rmsd_to_prob_transform(pred_act[:,2]).detach(), 
                    "target_pose":self.rmsd_to_prob_transform(batch_activity.predicted_rmsd).detach()
                    })
            
            self.log("nll_mean_term", torch.mean((batch_activity.y - pred_act[:,0]) ** 2 / variance), batch_size=batch_activity.num_graphs, on_step=True, on_epoch=True)
            self.log("regulariser", torch.mean(1 / variance), batch_size=batch_activity.num_graphs, on_step=True, on_epoch=True)
            self.log("nll_variance_term", torch.mean(torch.log(2 * torch.pi * variance)), batch_size= batch_activity.num_graphs, on_step=True,)

            

        if batch["pose"]:


            batch_pose = batch["pose"]
            

            # Forward pass for pose batch
            pred_pose = self.forward(batch_pose) 
            

            loss_pose = self.compute_loss_pose_nomask(pred_pose, batch_pose)
            self.log("train/loss_pose", loss_pose, batch_size=batch_pose.num_graphs, on_epoch=True, on_step=True)
            self.log("batch_pose", batch_pose.num_graphs, batch_size=batch_pose.num_graphs, on_epoch=False, on_step=True)
            
        
            # #pred_pose_prob=torch.sigmoid(pred_pose[:,2])
            pred_pose_prob=self.rmsd_to_prob_transform(pred_pose[:,2])
            target_rmsd=self.rmsd_to_prob_transform(batch_pose.predicted_rmsd)

            mae_training= (pred_pose_prob - target_rmsd).abs().mean() 
            self.log("pose_mae_training", mae_training, batch_size=batch_pose.num_graphs, on_epoch=True, on_step=True)


            pose_corr_training = ((pred_pose_prob - pred_pose_prob.mean()) * (target_rmsd - target_rmsd.mean())).mean() / (
            pred_pose_prob.std() * target_rmsd.std())

            self.log("pose_corr_training", pose_corr_training, batch_size=batch_pose.num_graphs, on_epoch=True, on_step=True)
            self.log("train/weight_pose", self.current_weight_pose, batch_size= batch_pose.num_graphs, on_epoch=True, on_step=False)

            self.training_step_outputs["pose"].append({
                    "pose_pred":pred_pose_prob.detach(), 
                    "target_pose":target_rmsd.detach()
                    })



        activity_loss = self.current_weight_pki * loss_activity if batch["activity"] is not None else 0
        pose_loss = self.current_weight_pose * loss_pose if batch["pose"] is not None else 0

        n_pose=batch["pose"].num_graphs if batch["pose"] is not None else 0
        n_act=batch["activity"].num_graphs

        self.log("batch_total", n_pose+n_act, batch_size=n_pose+n_act, on_epoch=True, on_step=True)

        #self.log("batch_total", n_act, batch_size=n_act, on_epoch=True, on_step=True) #del


        total_loss = activity_loss + pose_loss
        #total_loss = activity_loss #del

        self.log("train/total_loss", total_loss, batch_size= n_pose+n_act, on_epoch=True, on_step=True)
        #self.log("rmsd_shift", self.rmsd_shift, batch_size= n_pose+n_act, on_epoch=True, on_step=False)
        #self.log("rmsd_scale", self.rmsd_scale, batch_size= n_pose+n_act, on_epoch=True, on_step=True)
        #self.log("train/total_loss", total_loss, batch_size= n_act, on_epoch=True, on_step=True) #del




        return total_loss



    def on_train_epoch_end(self):

        #self.rmsd_shift.data.clamp(1.0, 10.0)
        #self.rmsd_scale.data.clamp(0.1, 5.0)
        #wandb.log({"rmsd_scale": self.rmsd_scale.item(), "rmsd_shift" : self.rmsd_shift.item()})
        

        #for q-q plot of trianing to compare to validation


        if self.current_epoch % 5 == 0 and self.training_step_outputs["activity"]:  # logging every 5 epochs

             import numpy as np
             import statsmodels.api as sm
             import matplotlib.pyplot as plt

             # Concatenate stored predictions, targets, and variances
             pred_activity = torch.cat([x["pred_activity"] for x in self.training_step_outputs["activity"]])
             target_activity = torch.cat([x["target_activity"] for x in self.training_step_outputs["activity"]])
             pred_variance = torch.cat([x["variance"] for x in self.training_step_outputs["activity"]])
             pose=torch.cat([x["pose"] for x in self.training_step_outputs["activity"]]).detach().cpu().numpy()
             target_pose = torch.cat([x["target_pose"] for x in self.training_step_outputs["activity"]]).detach().cpu().numpy()

             pose_batch=torch.cat([x["pose_pred"] for x in self.training_step_outputs["pose"]]).detach().cpu().numpy()
             target_pose_pose_batch=torch.cat([x["target_pose"] for x in self.training_step_outputs["pose"]]).detach().cpu().numpy()

             std_variance=torch.sqrt(pred_variance).detach().cpu().numpy()
             epoch=str(self.current_epoch)

             ####save the data
             target_activity = target_activity.detach().cpu().numpy()
             
             save_dir="/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/data_runs/"+str(self.directory_csv_name)

             pred_activity=pred_activity.detach().cpu().numpy()
             df = pd.DataFrame({"PredictedPoseQuality": pose, "PredictedStd": std_variance, "PredictedActivity": pred_activity, "TargetActivity": target_activity, "TargetPose":target_pose})
            
             # Define file name
             file_path = os.path.join(save_dir, f"training_{epoch}.csv")
            
             # Save to CSV
             df.to_csv(file_path, index=False)
            
             print(f"Saved calibration data at epoch {epoch} to {file_path}")
             
             ##Q-Q plot
             #name="training"
             # Compute residuals and standardised residuals
             #residuals = torch.abs(target_activity - pred_activity).detach().cpu().numpy()
             #y_std = torch.sqrt(pred_variance).detach().cpu().numpy()
             #qqplot=self.qqplot(residuals, y_std, name)

             ### kde plot unc act and pose
             
             #epoch=str(self.current_epoch)
             #fig=self.plot_unc_act_pose(pose, std_variance, epoch, name)


             #calibraiton plot unc activity and residuals
             #std_variance=torch.sqrt(pred_variance).detach().cpu().numpy()
             #cal = self.calibration_plot(std_variance, residuals, name)


             ## pose prediction distrib (davids dataset)
             #name_pose="pose_batch_training"
             #pose_plot = self.pose_distribution(pose_batch, name_pose)

             ## pose target distrib (davids dataset)
             #name_pose="target_pose_batch_training"
             #pose_plot = self.pose_distribution(target_pose_pose_batch, name_pose)




             ### kde plot act residuals and pose pred

             #name_2="training_act_residuals"
             
             #fig_act_res_pose=self.plot_unc_act_pose(pose, residuals, epoch, name_2)

             

        if self.current_epoch == 0:

            save_dir="/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/data_runs/"+str(self.directory_csv_name)

            df = pd.DataFrame({"TargetPose":target_pose_pose_batch})
            
            # Define file name
            file_path = os.path.join(save_dir, f"training_davids_data.csv")
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            print(f"Saved training pose data  to {file_path}")


        
 


        self.training_step_outputs.clear()

        

    def on_validation_epoch_start(self):
        """Reset validation storage at the start of each epoch."""
        self.validation_step_outputs = {"activity": [], "pose": []}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0, key: str = "val"):

        """Alternate validation based on dataset type (activity vs pose)."""

        #print(f"Batch index: {batch_idx}, Dataloader index: {dataloader_idx}, Batch type: {type(batch)}")

        

        if dataloader_idx == 0:  # Dataset 1 (Activity)
            n_act = batch.num_graphs
            
            
            # Forward pass for activity batch
            pred_activity = self.forward(batch)
            act_pred = pred_activity[:, 0]
            pred_log_variance = pred_activity[:, 1]
            pred_variance = torch.exp(pred_log_variance)
            #residuals = torch.abs(act_pred - batch.y)

            

            self.validation_step_outputs["activity"].append({
                "pred_activity": act_pred.detach(),
                "target_activity": batch.y.detach(),
                "variance": pred_variance.detach(),
                #"residuals": residuals.detach(),
                "pose":self.rmsd_to_prob_transform(pred_activity[:,2]).detach(),
                "target_pose":self.rmsd_to_prob_transform(batch.predicted_rmsd).detach()
                })

        elif dataloader_idx == 1:  # Dataset 2 (Pose)


            n_pose = batch.num_graphs


            # Forward pass for pose batch
            pred_pose_raw = self.forward(batch)
            target_exp_rmsd = batch.predicted_rmsd  # Raw RMSD target
            target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd)  # Transform to probability

            pred_pose_logit = pred_pose_raw[:, 2]
            pred_pose_prob = self.rmsd_to_prob_transform(pred_pose_logit)

            
            self.validation_step_outputs["pose"].append({
                "pred_pose": pred_pose_prob.detach(),
                "target_pose": target_rmsd.detach(),
                "scaffolds": batch.scaffold})



    def on_validation_epoch_end(self):
        """
        Computes validation metrics at epoch end, handling alternating datasets correctly.
        """


        activity_outputs = self.validation_step_outputs.get("activity", [])
        
        
        pred_activity = torch.cat([x["pred_activity"] for x in activity_outputs])
        target_activity = torch.cat([x["target_activity"] for x in activity_outputs])
        pred_variance = torch.cat([x["variance"] for x in activity_outputs])
        #residuals = torch.cat([x["residuals"] for x in activity_outputs])
        
        
        activity_mae = (pred_activity- target_activity).abs().mean()
            
        activity_corr = ((pred_activity - pred_activity.mean()) * (target_activity - target_activity.mean())).mean() / (
                pred_activity.std() * target_activity.std())


        self.log("val/mae_activity", activity_mae, on_epoch=True)
        self.log("val/corr_activity", activity_corr, on_epoch=True)

        if self.validation_step_outputs["pose"]:

            pose_outputs= self.validation_step_outputs.get("pose", [])

            pred_pose = torch.cat([x["pred_pose"] for x in pose_outputs])
            target_pose = torch.cat([x["target_pose"] for x in pose_outputs])
            
            pose_corr = ((pred_pose - pred_pose.mean()) * (target_pose - target_pose.mean())).mean() / (
                pred_pose.std() * target_pose.std())

            pose_mae = (pred_pose - target_pose).abs().mean()

            self.log("val/mae_pose", pose_mae, on_epoch=True)
            self.log("val/corr_pose", pose_corr, on_epoch=True)

        
            combined_mae = (activity_mae + pose_mae) / 2
            self.log("val/combined_mae", combined_mae, on_epoch=True)


         # Ensure we have pose data before running scaffold evaluation
        # if pred_pose is not None and target_pose is not None and scaffolds:
        #     dataframe = self.evaluate_per_scaffold(pred_pose, target_pose, scaffolds)

        #     # Plot scaffold-based evaluation metrics
        #     import matplotlib.pyplot as plt

        #     # Correlation per scaffold
        #     fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
        #     ax_corr.bar(dataframe["scaffold"], dataframe["correlation"], color='blue', alpha=0.7)
        #     ax_corr.set_xlabel('Scaffolds (Ordered by Number of Samples, from left to right)')
        #     plt.xticks(rotation=90, fontsize=8)
        #     ax_corr.set_ylabel('PC')  # Pearson Correlation
        #     ax_corr.set_title('Evaluations Per Scaffold Validation')
        #     plt.tight_layout()

        #     # MAE per scaffold
        #     fig_mae, ax_mae = plt.subplots(figsize=(12, 6))
        #     ax_mae.bar(dataframe["scaffold"], dataframe["mae"], color='blue', alpha=0.7)
        #     ax_mae.set_xlabel('Scaffolds (Ordered by Number of Samples, from left to right)')
        #     plt.xticks(rotation=90, fontsize=8)
        #     ax_mae.set_ylabel('MAE')  # Mean Absolute Error
        #     ax_mae.set_title('Evaluations Per Scaffold Validation MAE')
        #     plt.tight_layout()

        #     # Log both figures to Weights & Biases
        #     import wandb
        #     wandb.log({
        #         "evaluations_per_scaffold_validation_corr": wandb.Image(fig_corr),
        #         "evaluations_per_scaffold_validation_mae": wandb.Image(fig_mae)
        #     }, commit=False)

        #     # Close figures to avoid memory leaks
        #     plt.close(fig_corr)
        #     plt.close(fig_mae)


        #get first data iteration for val set

        if self.current_epoch == 0:

            save_dir="/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/data_runs/"+str(self.directory_csv_name)

            df = pd.DataFrame({"TargetPose":target_pose.detach().cpu().numpy()})
            
            # Define file name
            file_path = os.path.join(save_dir, f"validation_davids_data.csv")
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            print(f"Saved validation pose data  to {file_path}")

            


        #Generate Q-Q plot every 5 epochs
        if self.current_epoch % 5 == 0: #--> change to 5 later

            import numpy as np
            import statsmodels.api as sm
            import matplotlib.pyplot as plt

            target_activity = target_activity.detach().cpu().numpy() #if isinstance(target_activity, torch.Tensor) else target_activity
            pred_activity = pred_activity.detach().cpu().numpy() #if isinstance(pred_activity, torch.Tensor) else pred_activity
            variance = pred_variance.detach().cpu().numpy() #if isinstance(variance, torch.Tensor) else variance
            #residuals = np.abs(target_activity - pred_activity)

            pose_act=torch.cat([x["pose"] for x in activity_outputs]).detach().cpu().numpy()
            target_pose_act=torch.cat([x["target_pose"] for x in activity_outputs]).detach().cpu().numpy()

            std_variance = np.sqrt(variance)
            epoch=str(self.current_epoch)


            save_dir="/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/data_runs/"+str(self.directory_csv_name)

            df = pd.DataFrame({"PredictedPoseQuality": pose_act, "PredictedStd": std_variance, "PredictedActivity": pred_activity, "TargetActivity": target_activity, "TargetPose":target_pose_act})
            
            # Define file name
            file_path = os.path.join(save_dir, f"validation_{epoch}.csv")
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            print(f"Saved calibration validation data at epoch {epoch} to {file_path}")

            #residuals = residuals.detach().cpu().numpy()

            ##Q-Q plot
            #name="validation"
            #y_std = torch.sqrt(pred_variance).detach().cpu().numpy()
            #qqplot=self.qqplot(residuals, y_std, name)


            #calibraiton plot unc activity and residuals
            #std_variance = np.sqrt(variance)
            #cal = self.calibration_plot(std_variance, residuals, name)
            #self.calibration_plot(std_variance, residuals, name)

            ## pose prediction distrib (davids dataset)
            #name_pose="pose_batch"
            #pose_plot = self.pose_distribution(pred_pose.detach().cpu().numpy(), name_pose)

            ## pose prediction distrib (davids dataset)
            #name_pose="target_pose_batch"
            #pose_plot_target = self.pose_distribution(target_pose.detach().cpu().numpy(), name_pose)


            ### kde plot unc act and pose
             
            #epoch=str(self.current_epoch)
            #fig = self.plot_unc_act_pose(pose_act, std_variance, epoch, name)


            ### kde plot act residuals and pose pred            

            


        self.validation_step_outputs.clear()  # free memory


 

    # def predict_step(self, batch, *args):
        
    #     activity_batch, pose_batch = batch  # batch is a tuple from the DataLoader
    #     n_act, n_pose = activity_batch.num_graphs, pose_batch.num_graphs

         
    #     pred_act = self.forward(activity_batch)


    #     act_pred=pred_act[:, 0]
    #     log_unc_pred=pred_act[:, 1]
    #     variance=torch.exp(log_unc_pred)
        


    #     pred_pose_raw = self.forward(pose_batch)

    #     target_exp_rmsd = pose_batch.predicted_rmsd  # Assuming pose_batch has predicted_rmsd as a target

    #     target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd) # Transform target_exp_rmsd to target_rmsd
    #     #pred_pose = torch.sigmoid(pred_pose_raw[:, 2])
    #     pred_pose = self.rmsd_to_prob_transform(pred_pose_raw[:, 2])
        


    #     return {"pred activity": act_pred, "target": activity_batch.y, "pred unc activity": variance,  
	# 	        "pred pose": pred_pose, "pose_target": target_rmsd
    #             }
    #     #return { 
	# 	#        "pred pose": pred_pose, "pose_target": target_rmsd
    #     #        }

    def predict_step(self, batch, *args):
        """
        Predict step adapted for alternating dataset structure.
        """
        if isinstance(batch, tuple) and isinstance(batch[0], torch_geometric.data.Batch):
            batch = batch[0]  # Unpack tuple if needed

        if hasattr(batch, "y"):  # Activity dataset
            pred_act = self.forward(batch)
            act_pred = pred_act[:, 0]
            log_unc_pred = pred_act[:, 1]
            variance = torch.exp(log_unc_pred)

            return {
                "pred_activity": act_pred,
                "target_activity": batch.y,
                "pred_unc_activity": variance
            }

        elif hasattr(batch, "predicted_rmsd"):  # Pose dataset
            pred_pose_raw = self.forward(batch)
            target_exp_rmsd = batch.predicted_rmsd  
            target_rmsd = self.rmsd_to_prob_transform(target_exp_rmsd)  
            pred_pose = self.rmsd_to_prob_transform(pred_pose_raw[:, 2])

            return {
                "pred_pose": pred_pose,
                "target_pose": target_rmsd
            }

        else:
            raise ValueError("Batch does not match activity or pose dataset structure.")

        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, key="test")

    

    # def test_epoch_end(self, outputs, *args, **kwargs) -> None:
    #     #pred_activity, target_activity, pred_pose,target_pose, activity_corr, pose_corr, activity_mae, pose_mae = self.process_eval_outputs(outputs)
    #     pred_activity, variance, residuals, target_activity, pred_pose, target_pose, activity_corr, pose_corr, activity_mae, pose_mae, scaffolds = self.process_eval_outputs(outputs)
        
    #     self.log("test/mae_activity", activity_mae)
    #     self.log("test/corr_activity", activity_corr)
    #     self.log("test/mae_pose", pose_mae)
    #     self.log("test/corr_pose", pose_corr)

    #     if self.log_test_predictions:
    #         test_predictions = wandb.Artifact("test_predictions", type="predictions")
    #         data = cat_many(outputs, subset=["pred_activity", "target_activity", "pred_pose", "target_pose"])
    #         #data = cat_many(outputs, subset=["pred_pose", "target_pose"])
    #         values = [t.detach().cpu() for t in data.values()]
    #         values = torch.stack(values, dim=1)
    #         table = wandb.Table(columns=list(data.keys()), data=values.tolist())
    #         test_predictions.add(table, "predictions")
    #         wandb.log_artifact(test_predictions)
    #         pass

    def on_test_epoch_end(self) -> None:
        """
        Processes test outputs correctly when using alternating dataloaders.
        """
        
        # Flatten nested outputs from multiple dataloaders
        outputs = [item for sublist in outputs for item in sublist]

        # Process test outputs separately for activity and pose datasets
        (
            pred_activity, variance, residuals, target_activity, 
            pred_pose, target_pose, activity_corr, pose_corr, 
            activity_mae, pose_mae, scaffolds
        ) = self.process_eval_outputs(outputs)

        # Log individual test metrics
        if activity_mae is not None:
            self.log("test/mae_activity", activity_mae)
        if activity_corr is not None:
            self.log("test/corr_activity", activity_corr)
        if pose_mae is not None:
            self.log("test/mae_pose", pose_mae)
        if pose_corr is not None:
            self.log("test/corr_pose", pose_corr)

        # Log combined test MAE if both datasets exist
        if activity_mae is not None and pose_mae is not None:
            total_mae = (activity_mae + pose_mae) / 2
            self.log("test/combined_mae", total_mae)

        # Log test predictions to Weights & Biases
        if self.log_test_predictions:
            
            test_predictions = wandb.Artifact("test_predictions", type="predictions")

            # Gather only available predictions (avoid crashing if one dataset is missing)
            subset_keys = []
            if pred_activity is not None:
                subset_keys += ["pred_activity", "target_activity"]
            if pred_pose is not None:
                subset_keys += ["pred_pose", "target_pose"]

            # Collect and log the available outputs
            if subset_keys:
                data = cat_many(outputs, subset=subset_keys)
                values = [t.detach().cpu() for t in data.values()]
                values = torch.stack(values, dim=1)
                table = wandb.Table(columns=list(data.keys()), data=values.tolist())
                test_predictions.add(table, "predictions")
                wandb.log_artifact(test_predictions)


