import torch

# Load checkpoints
checkpoint_one_forward = torch.load("checkpoints_one_forward_test/model_all_ok_epoch_0.pt")
checkpoint_two_forward = torch.load("checkpoints_two_forward_test/model_all_ok_epoch_0.pt")

# Extract gradients from both checkpoints
gradients_one_forward = checkpoint_one_forward["gradients"]
gradients_two_forward = checkpoint_two_forward["gradients"]

for name in gradients_one_forward.keys():
    grad_one = gradients_one_forward[name]
    grad_two = gradients_two_forward[name]

    # Compute the absolute difference
    grad_diff = torch.norm(grad_one - grad_two)

    print(f"Gradient difference for {name}: {grad_diff.item()}")

