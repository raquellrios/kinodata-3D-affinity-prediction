import torch


before_one_pass=torch.load("weights_before_one_forward_pass.pt")
after_one_pass=torch.load("weights_after_one_forward_pass.pt")


# Load state_dicts for both models
#state_dict_one = model_one_forward.state_dict()
#state_dict_two = model_two_forward.state_dict()




for name in before_one_pass.keys():
    param_one = before_one_pass[name]
    param_two = after_one_pass[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch across forward pass")
        #print(f"Mismatch in {name}:")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")