import torch

state_dict_one= torch.load("checkpoints_initial_one_forward/initial_weights.pt")
state_dict_two = torch.load("checkpoints_initial_two_forward/initial_weights.pt")

# Load state_dicts for both models
#state_dict_one = model_one_forward.state_dict()
#state_dict_two = model_two_forward.state_dict()

# Compare weights and biases layer by layer
for name in state_dict_one.keys():
    param_one = state_dict_one[name]
    param_two = state_dict_two[name]
    print(param_one)
    print(param_two)
    
    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in {name}:")
        print(f"Model 1: {param_one}")
        print(f"Model 2: {param_two}")

