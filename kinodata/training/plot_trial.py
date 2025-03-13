import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🔹oad the saved CSV file
file_path = "data_runs/calibration_epoch_0_validation.csv"  # Change this path if needed
df = pd.read_csv(file_path)

# 🔹heck if the expected columns exist
if "PredictedPoseQuality" not in df.columns or "PredictedStd" not in df.columns:
    raise ValueError("CSV file does not contain expected columns: 'PoseQuality', 'PredictedStd'.")

#  Extract data
pose_quality = df["PredictedPoseQuality"].values
predicted_std = df["PredictedStd"].values
print("data loaded doing figure")
#  Create Figure
fig, ax = plt.subplots(figsize=(7, 5))

#  KDE Contour Plot
contour = sns.kdeplot(
    x=pose_quality,
    y=predicted_std,
    cmap="Blues",
    fill=True,
    levels=20,
    ax=ax
)

# Fix colorbar issue
mappable = contour.collections[-1]  # Get last contour for colorbar
fig.colorbar(mappable, ax=ax, label="Density")  # Attach the colorbar

#  Labels & Title
plt.xlabel("Pose Quality")
plt.ylabel("Predicted Uncertainty (σ)")
plt.title(f"Contour Plot of Uncertainty vs. Pose Quality (Epoch 10)")
plt.grid()

#  Save the figure to a local file
output_dir = "saved_plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "contour_plot_epoch_10.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved at: {output_path}")

# Show the figure
plt.show()










