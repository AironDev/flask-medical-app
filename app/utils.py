import nibabel as nib
import numpy as np
import torch
import base64
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO  # Import for base64 encoding of images
import torch.nn as nn
import torch.nn.functional as F
from flask import current_app
from io import BytesIO
import io

def preprocess_image(file):
    # Determine the file extension
    filename = file.filename
    ext = os.path.splitext(filename)[-1]  # Get the file extension (e.g., .nii, .nii.gz)

    # Define the path to the tmp directory
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)  # Ensure the tmp directory exists

    # Create a temporary file with the correct extension in the tmp directory
    with tempfile.NamedTemporaryFile(suffix=ext, dir=tmp_dir, delete=False) as tmp:
        file.save(tmp.name)
        temp_file_path = tmp.name

    try:
        # Load the image using nibabel
        image = nib.load(temp_file_path)
        image_data = image.get_fdata()

        # Check if the image has only one channel
        if image_data.ndim == 3:  # [D, H, W] -> Convert to [C, D, H, W]
            image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension

        # Repeat the single channel to match the expected input channels (5)
        input_data = np.repeat(image_data, 5, axis=0)  # Repeat along the channel axis
        
        # Convert the image data to a tensor
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    return input_tensor


def simulate_tumor_growth(shape=(240, 240, 160), num_iterations=100, diffusion_rate=0.1):
    # Initialize tumor tensor on CPU
    tumor = torch.zeros(shape, dtype=torch.float32, device='cpu')
    
    # Seed initial tumor at center
    tumor[shape[0]//2, shape[1]//2, shape[2]//2] = 1.0
    
    for _ in range(num_iterations):
        # Apply diffusion equation (simplified)
        tumor_new = tumor.clone()
        tumor_new[1:-1, 1:-1, 1:-1] += diffusion_rate * (
            tumor[:-2, 1:-1, 1:-1] + tumor[2:, 1:-1, 1:-1] +
            tumor[1:-1, :-2, 1:-1] + tumor[1:-1, 2:, 1:-1] +
            tumor[1:-1, 1:-1, :-2] + tumor[1:-1, 1:-1, 2:] -
            6 * tumor[1:-1, 1:-1, 1:-1]
        )
        
        # Clip values to [0, 1]
        tumor = torch.clamp(tumor_new, 0, 1)
    
    return tumor



def preprocess_images(zip_ref):
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract the zip file into the tmp directory
    zip_ref.extractall(tmp_dir)

    # List of required modalities
    required_modalities = ['flair', 't1ce', 't1', 't2']
    modality_files = {modality: None for modality in required_modalities}

    # Process each extracted file
    for file_name in os.listdir(tmp_dir):
        for modality in required_modalities:
            if file_name.endswith(f"_{modality}.nii") or file_name.endswith(f"_{modality}.nii.gz"):
                modality_files[modality] = os.path.join(tmp_dir, file_name)

    # Check if all required modalities are present
    if not all(modality_files.values()):
        raise ValueError('ZIP file must contain all four modalities: flair, t1ce, t1, t2')

    images = []
    for modality, file_path in modality_files.items():
        try:
            # Load the image using nibabel
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # Normalize and convert to float32
            img_data = (img_data - np.mean(img_data)) / np.std(img_data)
            images.append(img_data.astype(np.float32))

        except nib.filebasedimages.ImageFileError as e:
            print(f"Error loading image for modality {modality}: {e}")
            raise ValueError(f"Failed to load NIfTI image for modality '{modality}'. Please check the file format.")

    # Ensure all modalities have the same dimensions
    shapes = [img.shape for img in images]
    if len(set(shapes)) != 1:
        raise ValueError("All modalities must have the same dimensions")

    # Stack modalities into a single tensor
    image_stack = np.stack(images, axis=0)  # Stack along channel axis

    # Add a fifth dummy channel
    dummy_channel = np.zeros_like(image_stack[0])  # Create a zero-filled channel
    image_stack = np.concatenate([image_stack, np.expand_dims(dummy_channel, axis=0)], axis=0)

    image_tensor = torch.tensor(image_stack, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    return image_tensor





def inference(input_tensor):
    # Perform inference using the loaded model
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)  # Move input tensor to the correct device
    with torch.no_grad():
        output_tensor = model(input_tensor)  # Run inference
    return output_tensor

def visualize_results(image_tensor, output_tensor, filename="result_image.png"):
    # Convert tensors to NumPy arrays
    image_np = image_tensor[0].cpu().numpy()  # Convert input image to NumPy array
    output_np = output_tensor[0].cpu().numpy()  # Convert output to NumPy array

    # Determine the number of slices along the depth (D) dimension
    depth = image_np.shape[1]  # Assuming [C, D, H, W] format
    slice_idx = depth // 2  # Middle slice for visualization

    # Extract ET, TC, WT from the output tensor
    et_mask = output_np[0]  # Enhancing Tumor (ET)
    tc_mask = output_np[1]  # Tumor Core (TC)
    wt_mask = output_np[2]  # Whole Tumor (WT)

    # Create subplots to visualize input, output, and overlay side by side
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))

    # Plot input image channels
    for i in range(4):
        if i < image_np.shape[0]:  # Ensure within bounds
            axs[0, i].imshow(image_np[i, slice_idx, :, :], cmap="gray")
            axs[0, i].set_title(f"Input Channel {i+1}")
        else:
            axs[0, i].axis('off')  # Hide unused subplots

    # Plot output segmentation channels
    for i in range(4):
        if i < output_np.shape[0]:  # Ensure within bounds
            axs[1, i].imshow(output_np[i, slice_idx, :, :], cmap="gray")
            axs[1, i].set_title(f"Output Channel {i+1}")
        else:
            axs[1, i].axis('off')  # Hide unused subplots

    # Create overlay by combining input and output images
    for i in range(4):
        if i < image_np.shape[0] and i < output_np.shape[0]:  # Ensure within bounds
            overlay = np.copy(image_np[i, slice_idx, :, :])
            axs[2, i].imshow(overlay, cmap="gray", alpha=0.7)  # Input in grayscale
            axs[2, i].imshow(output_np[i, slice_idx, :, :], cmap="jet", alpha=0.3)  # Output in color
            axs[2, i].set_title(f"Overlay Channel {i+1}")
        else:
            axs[2, i].axis('off')  # Hide unused subplots

    # Plot overlays for each tumor class (ET, TC, WT)
    tumor_classes = [("Enhancing Tumor (ET)", et_mask), 
                     ("Tumor Core (TC)", tc_mask), 
                     ("Whole Tumor (WT)", wt_mask)]

    for j, (title, mask) in enumerate(tumor_classes):
        axs[3, j].imshow(image_np[0, slice_idx, :, :], cmap="gray", alpha=0.7)  # Input in grayscale
        axs[3, j].imshow(mask[slice_idx, :, :], cmap="jet", alpha=0.3)  # Tumor class in color
        axs[3, j].set_title(f"{title} Overlay")

    axs[3, 3].axis('off')  # Hide unused subplot

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  # 'static/uploads'
    os.makedirs(uploads_dir, exist_ok=True)  # Ensure the uploads directory exists

    # Save the figure to the uploads folder
    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename





def visualize_prediction(inputs, prediction, selected_slices=None):
    """
    Visualize the input modalities and the model's prediction.

    Parameters:
    - inputs: A 4D tensor or numpy array of shape (4, H, W, D), where each channel is a different modality.
    - prediction: The model's output, assumed to be a 4D tensor or numpy array of shape (C, H, W, D) 
                  where C is the number of classes.
    - selected_slices: A list of slice indices to visualize. If None, all slices are shown.
    """

    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()

    # Remove batch dimension if present
    inputs = inputs[0] if inputs.ndim == 5 else inputs
    prediction = prediction[0] if prediction.ndim == 5 else prediction

    selected_slices = [10, 30, 50, 70]

    # If selected slices are not provided, visualize all slices
    if selected_slices is None:
        selected_slices = range(inputs.shape[3])

    # Adjust the number of rows to accommodate all inputs and class predictions + overlay
    total_rows = 4 + prediction.shape[0] + 1  # 4 for input modalities, C for predictions, 1 for overlay
    fig, axes = plt.subplots(total_rows, len(selected_slices), figsize=(15, 8))

    # Plot input modalities (FLAIR, T1CE, T1, T2)
    for i, modality in enumerate(["FLAIR", "T1CE", "T1", "T2"]):
        for j, slice_idx in enumerate(selected_slices):
            axes[i, j].imshow(inputs[i, :, :, slice_idx], cmap='gray')
            axes[i, j].set_title(f"{modality} Slice {slice_idx}", fontsize=8)
            axes[i, j].axis('off')

    # Plot model predictions for each class (ET, WT, TC)
    classes = ["ET", "WT", "TC"]
    for class_idx in range(prediction.shape[0]):
        for j, slice_idx in enumerate(selected_slices):
            axes[4 + class_idx, j].imshow(prediction[class_idx, :, :, slice_idx], cmap='hot', alpha=0.5)
            axes[4 + class_idx, j].set_title(f"Prediction {classes[class_idx]} Slice {slice_idx}", fontsize=8)
            axes[4 + class_idx, j].axis('off')

    # Plot overlayed input and prediction
    for j, slice_idx in enumerate(selected_slices):
        overlay = np.zeros_like(inputs[0, :, :, slice_idx])
        for class_idx in range(prediction.shape[0]):
            overlay += prediction[class_idx, :, :, slice_idx] * (class_idx + 1)

        axes[4 + prediction.shape[0], j].imshow(inputs[0, :, :, slice_idx], cmap='gray')
        axes[4 + prediction.shape[0], j].imshow(overlay, cmap='jet', alpha=0.3)
        axes[4 + prediction.shape[0], j].set_title(f"Overlay Slice {slice_idx}", fontsize=8)
        axes[4 + prediction.shape[0], j].axis('off')

    plt.tight_layout(pad=2.0)

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  
    os.makedirs(uploads_dir, exist_ok=True)  

    # Save the figure to the uploads folder
    filename = "filename.png"
    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename






class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HybridSegResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters=16, dropout_prob=0.2):
        super(HybridSegResNetUNet, self).__init__()

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, init_filters)
        self.encoder2 = ResidualBlock(init_filters, init_filters * 2, stride=2)
        self.encoder3 = ResidualBlock(init_filters * 2, init_filters * 4, stride=2)
        self.encoder4 = ResidualBlock(init_filters * 4, init_filters * 8, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(init_filters * 8, init_filters * 16, stride=2)

        # Decoder
        self.decoder4 = nn.ConvTranspose3d(init_filters * 16, init_filters * 8, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose3d(init_filters * 8, init_filters * 4, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose3d(init_filters * 4, init_filters * 2, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose3d(init_filters * 2, init_filters, kernel_size=2, stride=2)

        # Final convolution
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d4 = self.decoder4(b)
        d4 = self._match_size(d4, e4)  # Ensure sizes match
        d4 = d4 + e4
        d4 = F.relu(d4)

        d3 = self.decoder3(d4)
        d3 = self._match_size(d3, e3)  # Ensure sizes match
        d3 = d3 + e3
        d3 = F.relu(d3)

        d2 = self.decoder2(d3)
        d2 = self._match_size(d2, e2)  # Ensure sizes match
        d2 = d2 + e2
        d2 = F.relu(d2)

        d1 = self.decoder1(d2)
        d1 = self._match_size(d1, e1)  # Ensure sizes match
        d1 = d1 + e1
        d1 = F.relu(d1)

        out = self.final_conv(d1)
        return out

    def _match_size(self, tensor, target):
        # Determine the sizes of the tensor and target
        target_size = target.size()
        current_size = tensor.size()
        
        # Calculate necessary padding or cropping
        pad_d = target_size[2] - current_size[2]
        pad_h = target_size[3] - current_size[3]
        pad_w = target_size[4] - current_size[4]
        
        # Apply padding if needed
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2]
            tensor = F.pad(tensor, padding, mode='constant', value=0)
        
        # Apply cropping if needed
        if pad_d < 0 or pad_h < 0 or pad_w < 0:
            tensor = tensor[..., :target_size[2], :target_size[3], :target_size[4]]
        
        return tensor

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained model here
    model = HybridSegResNetUNet(in_channels=5, out_channels=3).to(device)
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model
