import os
import torch
import tifffile
import numpy as np
from dataset import NucCorrDataset
from torchvision import transforms
from dataset import RandomShift3DWrapper

def save_samples(dataset, save_dir, num_samples=5):
    """
    Save sample images and segmentation masks from the dataset as tiff files.
    
    Args:
        dataset: NucCorrDataset instance
        save_dir: Directory to save the samples
        num_samples: Number of samples to save
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'incomplete_masks'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'complete_masks'), exist_ok=True)
    
    # Get random indices
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for i, idx in enumerate(indices):
        # Get sample
        incomplete_mask, image, complete_mask = dataset[idx]
        
        # Convert to numpy and remove channel dimension
        image = image.squeeze(0).numpy()
        incomplete_mask = incomplete_mask.squeeze(0).numpy()
        complete_mask = complete_mask.squeeze(0).numpy()
        
        # Convert image back to original range (0-255)
        image = (image * 255).astype(np.uint8)
        
        # Ensure masks are binary (0 or 1)
        incomplete_mask = (incomplete_mask > 0).astype(np.uint8)
        complete_mask = (complete_mask > 0).astype(np.uint8)
        
        # Save as tiff with appropriate metadata
        tifffile.imwrite(
            os.path.join(save_dir, 'images', f'sample_{i}_image.tiff'),
            image,
            compression='zlib',
            photometric='minisblack'
        )
        tifffile.imwrite(
            os.path.join(save_dir, 'incomplete_masks', f'sample_{i}_incomplete_mask.tiff'),
            incomplete_mask,
            compression='zlib',
            photometric='minisblack'
        )
        tifffile.imwrite(
            os.path.join(save_dir, 'complete_masks', f'sample_{i}_complete_mask.tiff'),
            complete_mask,
            compression='zlib',
            photometric='minisblack'
        )
        
        print(f'Saved sample {i}')
        print(f'Image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]')
        print(f'Incomplete mask shape: {incomplete_mask.shape}, dtype: {incomplete_mask.dtype}, range: [{incomplete_mask.min()}, {incomplete_mask.max()}]')
        print(f'Complete mask shape: {complete_mask.shape}, dtype: {complete_mask.dtype}, range: [{complete_mask.min()}, {complete_mask.max()}]')

if __name__ == '__main__':
    # Create dataset
    transform = transforms.Compose([
        RandomShift3DWrapper(max_shift=100),
        # Add other transforms here if needed
    ])
    
    train_dataset = NucCorrDataset(
        root_dir='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei',
        split_file='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_train.txt',
        transform=transform,
        train=True
    )
    
    # Save samples
    save_samples(
        dataset=train_dataset,
        save_dir='samples',
        num_samples=20
    ) 