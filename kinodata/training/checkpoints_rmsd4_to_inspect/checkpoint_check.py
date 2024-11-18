import torch
#healthy_checkpoint_5 = torch.load("model_all_ok_epoch_5.pt")  
#healthy_checkpoint_15 = torch.load("model_all_ok_epoch_15.pt")  
#healthy_checkpoint_20 = torch.load("model_all_ok_epoch_20.pt") 
healthy_checkpoint_20 = torch.load("model_all_ok_epoch_65.pt")   
nan_checkpoint = torch.load("model_nan_detected_epoch_68.pt") 

for name, param in healthy_checkpoint_20['model_state_dict'].items():
    nan_param = nan_checkpoint['model_state_dict'][name]
    # Check if there are NaNs in parameters
    if torch.isnan(nan_param).any():
        print(f"NaNs detected in parameter: {name}")

    # Calculate the difference between healthy and NaN parameters
    diff = (nan_param - param).abs()
    if torch.isnan(diff).any():
        print(f"NaNs in difference for parameter: {name}")
    elif diff.max() > 1e-3:  # Threshold for significant change
        print(f"Significant change in parameter {name}: max difference = {diff.max().item()}")


for name, grad in healthy_checkpoint_20['gradients'].items():
    nan_grad = nan_checkpoint['gradients'][name]
    # Check if there are NaNs in gradients
    if torch.isnan(nan_grad).any():
        print(f"NaNs detected in gradient: {name}")

    # Check for large changes in gradients
    grad_diff = (nan_grad - grad).abs()
    if torch.isnan(grad_diff).any():
        print(f"NaNs in gradient difference for parameter: {name}")
    elif grad_diff.max() > 1e-3:  # Threshold for significant change
        print(f"Significant change in gradient {name}: max difference = {grad_diff.max().item()}")




