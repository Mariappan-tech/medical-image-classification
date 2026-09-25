import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\maria\OneDrive\Desktop\madical image classification\data set\raw_data\pneumoniamnist_224.npz")

print("Keys in dataset:", list(data.keys()))

for key in data.keys():
    print(f"{key}: shape = {data[key].shape}, dtype = {data[key].dtype}")

train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']




print(f"\nTrain: {train_images.shape}, Labels: {train_labels.shape}")
print(f"Val:   {val_images.shape}, Labels: {val_labels.shape}")
print(f"Test:  {test_images.shape}, Labels: {test_labels.shape}")
print(f"\nLabel classes: {np.unique(train_labels)}")

plt.figure(figsize=(4, 4))
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0][0]}")
plt.axis('off')
plt.show()
