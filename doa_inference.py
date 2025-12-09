import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from pydantic import BaseModel, Field, field_validator
import warnings
warnings.filterwarnings("ignore")


# ==================== Configuration ====================
class InferenceConfig(BaseModel):
    """Configuration for DOA inference"""
    model_path: str = Field(..., description="Path to trained model checkpoint")
    device: Optional[str] = Field(None, description="Device to run inference on (cuda/cpu)")
    batch_size: int = Field(32, gt=0, description="Batch size for batch inference")
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Model file not found: {v}")
        return v


# ==================== Complex-Valued Operations ====================
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        real_out = self.conv_r(x.real) - self.conv_i(x.imag)
        imag_out = self.conv_r(x.imag) + self.conv_i(x.real)
        return torch.complex(real_out, imag_out)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_i = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        return torch.complex(self.bn_r(x.real), self.bn_i(x.imag))


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        real_out = self.fc_r(x.real) - self.fc_i(x.imag)
        imag_out = self.fc_r(x.imag) + self.fc_i(x.real)
        return torch.complex(real_out, imag_out)


class ComplexAdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    
    def forward(self, x):
        return torch.complex(self.pool(x.real), self.pool(x.imag))


class ComplexSEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = ComplexAdaptiveAvgPool1d(1)
        self.fc1 = ComplexLinear(channels, channels // reduction)
        self.fc2 = ComplexLinear(channels // reduction, channels)
        self.relu = ComplexReLU()
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        y_mag = torch.sqrt(y.real**2 + y.imag**2 + 1e-8)
        y_mag = torch.sigmoid(y_mag)
        y = torch.complex(y_mag * y.real, y_mag * y.imag)
        return x * y.view(b, c, 1)


# ==================== Model Architecture ====================
class ComplexCNNTransformer(nn.Module):
    def __init__(self, in_channels=2, embed_dim=256, num_heads=8, ff_dim=512, 
                 num_layers=2, max_len=512):
        super().__init__()
        
        self.conv1 = ComplexConv1d(in_channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = ComplexBatchNorm1d(16)
        self.conv2 = ComplexConv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = ComplexBatchNorm1d(32)
        self.conv3 = ComplexConv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = ComplexBatchNorm1d(64)
        self.conv4 = ComplexConv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = ComplexBatchNorm1d(128)
        self.conv5 = ComplexConv1d(128, 256, kernel_size=1, stride=2, padding=0)
        self.bn5 = ComplexBatchNorm1d(256)
        self.se5 = ComplexSEBlock1d(256)
        self.relu = ComplexReLU()
        
        self.complex_to_real_dim = 512
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, self.complex_to_real_dim))
        self.norm = nn.LayerNorm(self.complex_to_real_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.complex_to_real_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(self.complex_to_real_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.se5(self.relu(self.bn5(self.conv5(x))))
        
        b, c, w = x.size()
        x_real = torch.cat([x.real, x.imag], dim=1).permute(0, 2, 1)
        
        seq_len = x_real.size(1)
        x_real = x_real + self.pos_embed[:, :seq_len, :]
        x_real = self.norm(x_real)
        x_real = self.transformer(x_real).mean(dim=1)
        
        return self.fc(x_real).squeeze(-1)


# ==================== Inference Class ====================
class DOAInference:
    
    def __init__(self, model_path: str, device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize the DOA model
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Default batch size for batch_predict
        """
        # Validate configuration
        self.config = InferenceConfig(
            model_path=model_path,
            device=device,
            batch_size=batch_size
        )
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize and load model
        self.model = self._load_model()
        
        print(f"✓ DOA Inference initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load the trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.config.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Initialize model
            model = ComplexCNNTransformer(in_channels=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Print model info if available
            if 'best_rmse' in checkpoint:
                print(f"✓ Model RMSE: {checkpoint['best_rmse']:.2f}°")
            if 'epoch' in checkpoint:
                print(f"✓ Trained epochs: {checkpoint['epoch'] + 1}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _normalize_iq_minmax(self, complex_data: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to I and Q components separately
        
        Args:
            complex_data: Complex numpy array
        
        Returns:
            Normalized complex data with I and Q components in range [0, 1]
        """
        i_component = np.real(complex_data)
        q_component = np.imag(complex_data)
        
        i_min, i_max = np.min(i_component), np.max(i_component)
        q_min, q_max = np.min(q_component), np.max(q_component)
        
        if i_max != i_min:
            i_normalized = (i_component - i_min) / (i_max - i_min)
        else:
            i_normalized = np.zeros_like(i_component)
        
        if q_max != q_min:
            q_normalized = (q_component - q_min) / (q_max - q_min)
        else:
            q_normalized = np.zeros_like(q_component)
        
        normalized_data = i_normalized + 1j * q_normalized
        return normalized_data
        
    def _preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data to complex tensor
        
        Args:
            data: Input array of shape (channels, samples) containing complex IQ data
            
        Returns:
            Complex tensor ready for model input
        """
        if data.dtype != np.complex64 and data.dtype != np.complex128:
            raise ValueError("Input data must be complex (IQ data)")
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D input (channels, samples), got shape {data.shape}")
        
        # Apply min-max normalization
        normalized_data = self._normalize_iq_minmax(data)
        
        return torch.tensor(normalized_data, dtype=torch.complex64)
    
    def predict(self, data: np.ndarray) -> float:
        """
        Predict DOA for a single sample
        
        Args:
            data: Complex IQ data of shape (channels, samples)
            
        Returns:
            DOA angle in degrees (0-180°)
        """
        try:
            # Preprocess
            iq_data = self._preprocess(data)
            iq_data = iq_data.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Inference
            with torch.no_grad():
                pred = self.model(iq_data)
                doa_deg = pred.cpu().item() * 180.0
            
            return doa_deg
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, data_list: List[np.ndarray], 
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict DOA for a batch of samples
        
        Args:
            data_list: List of complex IQ data arrays, each of shape (channels, samples)
            batch_size: Batch size for processing (uses default if None)
            
        Returns:
            Array of DOA angles in degrees (0-180°)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        predictions = []
        failed_indices = []
        
        try:
            # Process in batches
            for i in range(0, len(data_list), batch_size):
                batch_data = data_list[i:i+batch_size]
                batch_tensors = []
                
                # Preprocess batch
                for idx, data in enumerate(batch_data):
                    try:
                        tensor = self._preprocess(data)
                        batch_tensors.append(tensor)
                    except Exception as e:
                        print(f"⚠ Warning: Failed to process sample {i+idx}: {str(e)}")
                        failed_indices.append(i + idx)
                        batch_tensors.append(None)
                
                # Filter out failed samples
                valid_tensors = [t for t in batch_tensors if t is not None]
                
                if not valid_tensors:
                    predictions.extend([np.nan] * len(batch_data))
                    continue
                
                # Stack and predict
                batch_tensor = torch.stack(valid_tensors).to(self.device)
                
                with torch.no_grad():
                    preds = self.model(batch_tensor)
                    preds_deg = preds.cpu().numpy() * 180.0
                
                # Insert predictions (including NaN for failed samples)
                pred_idx = 0
                for tensor in batch_tensors:
                    if tensor is not None:
                        predictions.append(preds_deg[pred_idx])
                        pred_idx += 1
                    else:
                        predictions.append(np.nan)
            
            if failed_indices:
                print(f"⚠ Warning: {len(failed_indices)} samples failed processing")
            
            return np.array(predictions)
            
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def __repr__(self):
        return (f"DOAInference(device={self.device}, "
                f"batch_size={self.config.batch_size})")


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Initialize inference
    doa_model = DOAInference(
        model_path="cvnn_doa_best_model.pth",
        device=None,
        batch_size=32
    )
    
    sample_data = np.load("sample_data.npy")
    
    # Single sample inference
    doa = doa_model.predict(sample_data)
    print("Predicted DOA : ", doa)
    
    batch_data = [sample_data for i in range(16)]
    doas = doa_model.batch_predict(batch_data)
    print("Batch DOA Predictions : ", doas)
    