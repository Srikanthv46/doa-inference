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
class ComplexLinear(nn.Module):
    """Complex-valued linear transformation"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        real_part = self.fc_real(x.real) - self.fc_imag(x.imag)
        imag_part = self.fc_real(x.imag) + self.fc_imag(x.real)
        return torch.complex(real_part, imag_part)


class ComplexLayerNorm(nn.Module):
    """Complex-valued Layer Normalization"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_imag = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x):
        real_normalized = self.ln_real(x.real)
        imag_normalized = self.ln_imag(x.imag)
        return torch.complex(real_normalized, imag_normalized)


class ComplexBatchNorm1d(nn.Module):
    """Complex-valued Batch Normalization for 1D data"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        real_normalized = self.bn_real(x.real)
        imag_normalized = self.bn_imag(x.imag)
        return torch.complex(real_normalized, imag_normalized)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        real_part = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag_part = self.conv_real(x.imag) + self.conv_imag(x.real)
        return torch.complex(real_part, imag_part)


class ComplexReLU(nn.Module):
    """Complex ReLU: applies ReLU to both real and imaginary parts"""
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexSEBlock(nn.Module):
    """Complex-valued Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = ComplexLinear(channels, channels // reduction, bias=False)
        self.fc2 = ComplexLinear(channels // reduction, channels, bias=False)
        self.relu = ComplexReLU()
    
    def forward(self, x):
        b, c, seq_len = x.size()
        y = x.mean(dim=2)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        magnitude = torch.sqrt(y.real ** 2 + y.imag ** 2 + 1e-8)
        phase = torch.atan2(y.imag, y.real)
        weight = torch.sigmoid(magnitude)
        y = weight * torch.complex(torch.cos(phase), torch.sin(phase))
        return x * y.unsqueeze(-1)


class ComplexMultiHeadAttention(nn.Module):
    """Complex-valued Multi-Head Attention"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = ComplexLinear(embed_dim, embed_dim)
        self.k_proj = ComplexLinear(embed_dim, embed_dim)
        self.v_proj = ComplexLinear(embed_dim, embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        K_H = K.conj().transpose(-2, -1)
        
        attn_scores_real = torch.matmul(Q.real, K_H.real) - torch.matmul(Q.imag, K_H.imag)
        attn_scores_imag = torch.matmul(Q.real, K_H.imag) + torch.matmul(Q.imag, K_H.real)
        attn_scores = torch.complex(attn_scores_real, attn_scores_imag)
        
        attn_scores = attn_scores * self.scale
        
        attn_weights = torch.abs(attn_scores)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output_real = torch.matmul(attn_weights, V.real)
        output_imag = torch.matmul(attn_weights, V.imag)
        output = torch.complex(output_real, output_imag)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output


class ComplexTransformerEncoderLayer(nn.Module):
    """Complex-valued Transformer Encoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.self_attn = ComplexMultiHeadAttention(d_model, nhead)
        
        self.linear1 = ComplexLinear(d_model, dim_feedforward)
        self.linear2 = ComplexLinear(dim_feedforward, d_model)
        
        self.norm1 = ComplexLayerNorm(d_model)
        self.norm2 = ComplexLayerNorm(d_model)
        
        self.activation = ComplexReLU()
    
    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_output)
        
        return x


# ==================== Model Architecture ====================
class ComplexCNNTransformer(nn.Module):
    """Complex-Valued CNN-Transformer for DOA Estimation"""
    def __init__(self, in_channels=2, embed_dim=256, num_heads=8, ff_dim=512, 
                 num_layers=2, max_len=512):
        super().__init__()
        
        # Complex CNN layers
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
        self.se5 = ComplexSEBlock(256)
        
        self.activation = ComplexReLU()
        
        # Positional embedding
        self.pos_embed_real = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.pos_embed_imag = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.norm = ComplexLayerNorm(embed_dim)
        
        # Complex Transformer Encoder
        self.transformer_layers = nn.ModuleList([
            ComplexTransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection (complex to real)
        self.fc_real = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.se5(self.activation(self.bn5(self.conv5(x))))
        
        # Reshape for transformer
        x = x.transpose(1, 2)
        
        # Add positional embedding
        pos_embed = torch.complex(
            self.pos_embed_real[:, :x.size(1), :], 
            self.pos_embed_imag[:, :x.size(1), :]
        )
        x = x + pos_embed
        x = self.norm(x)
        
        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Project to real output
        x_magnitude = torch.sqrt(x.real ** 2 + x.imag ** 2)
        output = torch.sigmoid(self.fc_real(x_magnitude)).squeeze(-1)
        
        return output


# ==================== Inference Class ====================
class DOAInference:
    """
    Direction of Arrival (DOA) Inference Class
    
    Loads a trained Complex-Valued CNN-Transformer model and performs inference
    on single samples or batches of samples.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize the DOA inference class
        
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
        Preprocess input data to complex tensor with normalization
        
        Args:
            data: Input array of shape (channels, samples) containing complex IQ data
            
        Returns:
            Normalized complex tensor ready for model input
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
    inference = DOAInference(
        model_path="doa_model.pth",
        device=None,
        batch_size=32
    )
    
    sample_data = np.load("sample_data.npy") # IQ data shape (2 channels, 4096 samples)
    sample_data = sample_data.astype(np.complex64) 
    
    # Streaming inference
    doa = inference.predict(sample_data)
    print(f"\nSingle prediction: {doa:.2f}°")
    
    # Batch prediction
    batch_data = [
        sample_data
        for _ in range(10)
    ]
    batch_data = [d.astype(np.complex64) for d in batch_data]
    
    doas = inference.batch_predict(batch_data)
    print(f"\nBatch predictions: {doas}")
    print(f"Median DOA: {np.nanmedian(doas):.2f}°")