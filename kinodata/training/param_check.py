import torch

before_val_one= torch.load("checkpoints_one_forward_test/one_forward_before_validation.pt")
before_val_two= torch.load("checkpoints_two_forward_test/two_forward_before_validation.pt")
after_val_one= torch.load("checkpoints_one_forward_test/one_forward_after_validation.pt")
after_val_two= torch.load("checkpoints_two_forward_test/two_forward_after_validation.pt")
before_init_train_one= torch.load("checkpoints_one_forward_test/one_forward_before_initial_training.pt")
before_init_train_two= torch.load("checkpoints_two_forward_test/two_forward_before_initial_training.pt")
before_train_one= torch.load("checkpoints_one_forward_test/one_forward_before_training.pt")
before_train_two= torch.load("checkpoints_two_forward_test/two_forward_before_training.pt")
#after_train_one= torch.load("checkpoints_one_forward_test/one_forward_after_training.pt")
#after_train_two= torch.load("checkpoints_two_forward_test/two_forward_after_training.pt")


# Load state_dicts for both models
#state_dict_one = model_one_forward.state_dict()
#state_dict_two = model_two_forward.state_dict()



# Compare weights and biases layer by layer
for name in before_val_one.keys():
    mismatch_found = False
    param_one = before_val_one[name]
    param_two = before_val_two[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in before val")
        mismatch_found = True  # Set the flag
        break 
        #print(f"Mismatch in {name}:")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")


for name in after_val_one.keys():
    mismatch_found = False
    param_one = after_val_one[name]
    param_two = after_val_two[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in  after val")
        mismatch_found = True  # Set the flag
        break 
        #print(f"Mismatch in {name} in after val")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")



for name in before_init_train_one.keys():
    mismatch_found = False
    param_one = before_init_train_one[name]
    param_two = before_init_train_two[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in before train init")
        mismatch_found = True  # Set the flag
        break 
        #print(f"Mismatch in {name} in before train init")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")


for name in before_train_one.keys():
    mismatch_found = False
    param_one = before_train_one[name]
    param_two = before_train_two[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in before train init ")
        mismatch_found = True  # Set the flag
        break
        #print(f"Mismatch in {name} in before train ")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")


for name in before_train_one.keys():
    mismatch_found = False
    param_one = before_train_one[name]
    param_two = before_init_train_one[name]
    #print(param_one)
    #print(param_two)

    if not torch.allclose(param_one, param_two, atol=1e-6):  # Adjust tolerance as needed
        print(f"Mismatch in before train init and first training for one pass")
        mismatch_found = True  # Set the flag
        break
        #print(f"Mismatch in {name} in before train ")
        #print(f"Model 1: {param_one}")
        #print(f"Model 2: {param_two}")


