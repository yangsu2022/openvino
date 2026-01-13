#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
MoE Expert Weights Extractor v2

This tool extracts MoE (Mixture of Experts) expert weights from an OpenVINO IR model
and saves them to a binary file that can be used for OTD (Offload-To-Disk) feature.

For gpt-oss-20b-int4 model:
- 24 MoE layers
- 32 experts per layer
- Weights are INT4 compressed with scales

The output file format:
- Header (128 bytes):
  - magic: "MOEW" (4 bytes)
  - version: uint32 (4 bytes)
  - num_layers: uint32 (4 bytes)
  - num_experts_per_layer: uint32 (4 bytes)
  - expert_up_weight_size: uint64 (8 bytes)
  - expert_down_weight_size: uint64 (8 bytes)
  - expert_up_scale_size: uint64 (8 bytes)
  - expert_down_scale_size: uint64 (8 bytes)
  - expert_up_bias_size: uint64 (8 bytes)
  - expert_down_bias_size: uint64 (8 bytes)
  - data_offset: uint64 (8 bytes)
  - reserved: uint64[8] (64 bytes)

- Per-Layer Data:
  For each layer, for each expert:
    - up_weight
    - up_scale
    - up_bias
    - down_weight
    - down_scale
    - down_bias

Usage:
    python extract_moe_weights_v2.py --model <path_to_model_dir> --output <output_file>
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

try:
    import openvino as ov
except ImportError:
    print("Error: openvino package not found. Please install it with: pip install openvino")
    exit(1)


# Header format
HEADER_MAGIC = b'MOEW'
HEADER_VERSION = 1
HEADER_SIZE = 128  # Total header size in bytes

# Model-specific constants for gpt-oss-20b-int4
DEFAULT_NUM_LAYERS = 24
DEFAULT_NUM_EXPERTS = 32


class MoEWeightExtractor:
    """Extracts MoE expert weights from OpenVINO IR model."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.core = ov.Core()
        self.model = None
        
        # MoE structure info
        self.num_layers = 0
        self.num_experts = 0
        
        # Per-expert sizes in bytes
        self.expert_up_weight_size = 0
        self.expert_down_weight_size = 0
        self.expert_up_scale_size = 0
        self.expert_down_scale_size = 0
        self.expert_up_bias_size = 0
        self.expert_down_bias_size = 0
        
        # Weight data storage: layer_idx -> weight_type -> numpy array [num_experts, ...]
        self.weights_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
        
    def load_model(self) -> bool:
        """Load the OpenVINO IR model."""
        model_xml = self.model_path / "openvino_model.xml"
        if not model_xml.exists():
            print(f"Error: Model file not found: {model_xml}")
            return False
        
        print(f"Loading model from: {model_xml}")
        self.model = self.core.read_model(str(model_xml))
        print(f"Model loaded successfully. Total nodes: {len(self.model.get_ops())}")
        return True
    
    def analyze_moe_structure(self) -> bool:
        """Analyze the model to find MoE layers and their structure."""
        if self.model is None:
            return False
        
        print("\nAnalyzing MoE structure...")
        
        # Find all constants with [32, ...] shape (32 experts)
        expert_constants = []
        layer_indices = set()
        
        for op in self.model.get_ops():
            if op.get_type_name() != "Constant":
                continue
                
            name = op.get_friendly_name()
            shape = list(op.get_output_tensor(0).shape)
            
            # Look for expert-related tensors with 32 as first dim
            if len(shape) >= 2 and shape[0] == 32:
                # Extract layer index
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    layer_indices.add(layer_idx)
                    expert_constants.append((name, shape, layer_idx))
        
        if not layer_indices:
            print("Warning: Could not find expert layer indices from model")
            self.num_layers = DEFAULT_NUM_LAYERS
            self.num_experts = DEFAULT_NUM_EXPERTS
        else:
            self.num_layers = max(layer_indices) + 1
            self.num_experts = DEFAULT_NUM_EXPERTS
        
        print(f"\nMoE structure detected:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Experts per layer: {self.num_experts}")
        print(f"  Expert-related constants found: {len(expert_constants)}")
        
        return True
    
    def extract_weights(self) -> bool:
        """Extract expert weights from the model."""
        if self.model is None:
            return False
        
        print("\nExtracting expert weights...")
        
        # Key shapes for gpt-oss-20b-int4:
        #   [32, 5760, 90, 32] - up weights (INT4)
        #   [32, 5760, 90, 1]  - up scales
        #   [32, 1, 5760]      - up bias
        #   [32, 2880, 90, 32] - down weights (INT4)
        #   [32, 2880, 90, 1]  - down scales
        #   [32, 1, 2880]      - down bias
        
        up_weight_shape = (32, 5760, 90, 32)    # INT4 packed
        up_scale_shape = (32, 5760, 90, 1)
        up_bias_shape = (32, 1, 5760)
        down_weight_shape = (32, 2880, 90, 32)  # INT4 packed
        down_scale_shape = (32, 2880, 90, 1)
        down_bias_shape = (32, 1, 2880)
        
        # Storage for weights by layer
        self.weights_by_layer = {i: {} for i in range(self.num_layers)}
        
        # Collect tensors by shape (name doesn't contain layer info for weights)
        tensors_by_shape: Dict[tuple, List[Tuple[str, Any]]] = {
            up_weight_shape: [],
            up_scale_shape: [],
            up_bias_shape: [],
            down_weight_shape: [],
            down_scale_shape: [],
            down_bias_shape: [],
        }
        
        for op in self.model.get_ops():
            if op.get_type_name() != "Constant":
                continue
            
            name = op.get_friendly_name()
            shape = tuple(op.get_output_tensor(0).shape)
            
            if shape in tensors_by_shape:
                tensors_by_shape[shape].append((name, op))
        
        # Sort each group by name to get consistent layer ordering
        for shape in tensors_by_shape:
            tensors_by_shape[shape].sort(key=lambda x: x[0])
        
        # Map shapes to weight types
        shape_to_type = {
            up_weight_shape: 'up_weight',
            up_scale_shape: 'up_scale', 
            up_bias_shape: 'up_bias',
            down_weight_shape: 'down_weight',
            down_scale_shape: 'down_scale',
            down_bias_shape: 'down_bias',
        }
        
        found_count = {'up_weight': 0, 'up_scale': 0, 'up_bias': 0,
                       'down_weight': 0, 'down_scale': 0, 'down_bias': 0}
        
        # Extract weights for each shape type
        for shape, tensors in tensors_by_shape.items():
            weight_type = shape_to_type[shape]
            
            for layer_idx, (name, op) in enumerate(tensors):
                if layer_idx >= self.num_layers:
                    break
                
                try:
                    data = op.get_data()
                    self.weights_by_layer[layer_idx][weight_type] = data
                    found_count[weight_type] += 1
                except Exception as e:
                    print(f"Warning: Failed to get data for {name}: {e}")
        
        # Report what was found
        print("\nWeight extraction results:")
        for wtype, count in found_count.items():
            print(f"  {wtype}: {count}/{self.num_layers} layers")
        
        # Calculate per-expert sizes from first layer
        # Note: INT4 weights are packed as 1D int8 array, so we divide by num_experts
        if 0 in self.weights_by_layer:
            layer0 = self.weights_by_layer[0]
            if 'up_weight' in layer0:
                # For INT4 packed data, total_bytes / num_experts
                total_bytes = layer0['up_weight'].nbytes
                self.expert_up_weight_size = total_bytes // self.num_experts
                print(f"\n  up_weight per expert: {self.expert_up_weight_size} bytes ({self.expert_up_weight_size/1024/1024:.2f} MB)")
            if 'up_scale' in layer0:
                # Scale data has proper shape [32, ...]
                if len(layer0['up_scale'].shape) > 1:
                    self.expert_up_scale_size = layer0['up_scale'][0].nbytes
                else:
                    self.expert_up_scale_size = layer0['up_scale'].nbytes // self.num_experts
                print(f"  up_scale per expert: {self.expert_up_scale_size} bytes ({self.expert_up_scale_size/1024:.2f} KB)")
            if 'up_bias' in layer0:
                if len(layer0['up_bias'].shape) > 1:
                    self.expert_up_bias_size = layer0['up_bias'][0].nbytes
                else:
                    self.expert_up_bias_size = layer0['up_bias'].nbytes // self.num_experts
                print(f"  up_bias per expert: {self.expert_up_bias_size} bytes ({self.expert_up_bias_size/1024:.2f} KB)")
            if 'down_weight' in layer0:
                total_bytes = layer0['down_weight'].nbytes
                self.expert_down_weight_size = total_bytes // self.num_experts
                print(f"  down_weight per expert: {self.expert_down_weight_size} bytes ({self.expert_down_weight_size/1024/1024:.2f} MB)")
            if 'down_scale' in layer0:
                if len(layer0['down_scale'].shape) > 1:
                    self.expert_down_scale_size = layer0['down_scale'][0].nbytes
                else:
                    self.expert_down_scale_size = layer0['down_scale'].nbytes // self.num_experts
                print(f"  down_scale per expert: {self.expert_down_scale_size} bytes ({self.expert_down_scale_size/1024:.2f} KB)")
            if 'down_bias' in layer0:
                if len(layer0['down_bias'].shape) > 1:
                    self.expert_down_bias_size = layer0['down_bias'][0].nbytes
                else:
                    self.expert_down_bias_size = layer0['down_bias'].nbytes // self.num_experts
                print(f"  down_bias per expert: {self.expert_down_bias_size} bytes ({self.expert_down_bias_size/1024:.2f} KB)")
        
        # Check if we found enough weights
        total_found = sum(found_count.values())
        if total_found == 0:
            print("\nWarning: No weights found with expected shapes.")
            print("Will create file with simulated structure.")
            return False
        
        return True
    
    def _get_expert_data(self, layer_data: dict, weight_type: str, expert_idx: int, expert_size: int) -> bytes:
        """Extract data for a single expert from the layer data."""
        if weight_type not in layer_data:
            return b'\x00' * expert_size if expert_size > 0 else b''
        
        data = layer_data[weight_type]
        
        # Check if data is 1D (packed INT4) or multi-dimensional
        if len(data.shape) == 1:
            # 1D packed data - slice by byte offset
            start = expert_idx * expert_size
            end = start + expert_size
            return data[start:end].tobytes()
        else:
            # Multi-dimensional data - index by expert
            return data[expert_idx].tobytes()
    
    def write_otd_file(self, output_path: str) -> bool:
        """Write the extracted weights to OTD binary file."""
        print(f"\nWriting OTD file: {output_path}")
        
        with open(output_path, 'wb') as f:
            # Write header (128 bytes total)
            # 4s + III + 6Q + Q + 7Q = 4 + 12 + 48 + 8 + 56 = 128 bytes
            header = struct.pack(
                '<4sIII QQQQQQ Q 7Q',
                HEADER_MAGIC,
                HEADER_VERSION,
                self.num_layers,
                self.num_experts,
                self.expert_up_weight_size,
                self.expert_down_weight_size,
                self.expert_up_scale_size,
                self.expert_down_scale_size,
                self.expert_up_bias_size,
                self.expert_down_bias_size,
                HEADER_SIZE,  # data_offset
                0, 0, 0, 0, 0, 0, 0  # reserved (7 x uint64)
            )
            assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} != {HEADER_SIZE}"
            f.write(header)
            
            # Write data: layer by layer, expert by expert
            bytes_written = 0
            for layer_idx in range(self.num_layers):
                layer_data = self.weights_by_layer.get(layer_idx, {})
                
                for expert_idx in range(self.num_experts):
                    # Write up_weight
                    data = self._get_expert_data(layer_data, 'up_weight', expert_idx, self.expert_up_weight_size)
                    f.write(data)
                    bytes_written += len(data)
                    
                    # Write up_scale
                    data = self._get_expert_data(layer_data, 'up_scale', expert_idx, self.expert_up_scale_size)
                    f.write(data)
                    bytes_written += len(data)
                    
                    # Write up_bias
                    data = self._get_expert_data(layer_data, 'up_bias', expert_idx, self.expert_up_bias_size)
                    f.write(data)
                    bytes_written += len(data)
                    
                    # Write down_weight
                    data = self._get_expert_data(layer_data, 'down_weight', expert_idx, self.expert_down_weight_size)
                    f.write(data)
                    bytes_written += len(data)
                    
                    # Write down_scale
                    data = self._get_expert_data(layer_data, 'down_scale', expert_idx, self.expert_down_scale_size)
                    f.write(data)
                    bytes_written += len(data)
                    
                    # Write down_bias
                    data = self._get_expert_data(layer_data, 'down_bias', expert_idx, self.expert_down_bias_size)
                    f.write(data)
                    bytes_written += len(data)
                
                # Progress update
                if (layer_idx + 1) % 8 == 0 or layer_idx == self.num_layers - 1:
                    print(f"  Progress: {layer_idx + 1}/{self.num_layers} layers, {bytes_written/1024/1024/1024:.2f} GB written")
        
        file_size = os.path.getsize(output_path)
        print(f"\nOTD file created successfully:")
        print(f"  File size: {file_size/1024/1024/1024:.2f} GB")
        print(f"  Layers: {self.num_layers}")
        print(f"  Experts per layer: {self.num_experts}")
        
        return True


def create_test_otd_file(output_path: str, num_layers: int = 24, num_experts: int = 32):
    """Create a test OTD file with correct sizes based on gpt-oss-20b-int4."""
    print(f"Creating test OTD file: {output_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    
    # Actual sizes from gpt-oss-20b-int4 model analysis:
    # INT4 weights: [5760, 90, 32] = 8,294,400 int4 values = 4,147,200 bytes
    # But since INT4 is packed, we need to calculate correctly
    
    # Up projection: [5760, 90, 32] INT4 = 5760 * 90 * 32 / 2 = 8,294,400 bytes
    expert_up_weight_size = 5760 * 90 * 32 // 2  # INT4 packed
    # Down projection: [2880, 90, 32] INT4 = 2880 * 90 * 32 / 2 = 4,147,200 bytes  
    expert_down_weight_size = 2880 * 90 * 32 // 2  # INT4 packed
    
    # Scales: [5760, 90, 1] FP16 = 5760 * 90 * 2 = 1,036,800 bytes
    expert_up_scale_size = 5760 * 90 * 2  # FP16
    # Scales: [2880, 90, 1] FP16 = 2880 * 90 * 2 = 518,400 bytes
    expert_down_scale_size = 2880 * 90 * 2  # FP16
    
    # Bias: [1, 5760] FP32 = 5760 * 4 = 23,040 bytes
    expert_up_bias_size = 5760 * 4  # FP32
    # Bias: [1, 2880] FP32 = 2880 * 4 = 11,520 bytes
    expert_down_bias_size = 2880 * 4  # FP32
    
    per_expert_size = (expert_up_weight_size + expert_down_weight_size +
                       expert_up_scale_size + expert_down_scale_size +
                       expert_up_bias_size + expert_down_bias_size)
    total_size = HEADER_SIZE + per_expert_size * num_experts * num_layers
    
    print(f"\nPer-expert sizes:")
    print(f"  up_weight: {expert_up_weight_size/1024/1024:.2f} MB")
    print(f"  up_scale: {expert_up_scale_size/1024:.2f} KB")
    print(f"  up_bias: {expert_up_bias_size/1024:.2f} KB")
    print(f"  down_weight: {expert_down_weight_size/1024/1024:.2f} MB")
    print(f"  down_scale: {expert_down_scale_size/1024:.2f} KB")
    print(f"  down_bias: {expert_down_bias_size/1024:.2f} KB")
    print(f"  Total per expert: {per_expert_size/1024/1024:.2f} MB")
    print(f"\nExpected file size: {total_size/1024/1024/1024:.2f} GB")
    
    with open(output_path, 'wb') as f:
        # Write header (128 bytes total)
        header = struct.pack(
            '<4sIII QQQQQQ Q 7Q',
            HEADER_MAGIC,
            HEADER_VERSION,
            num_layers,
            num_experts,
            expert_up_weight_size,
            expert_down_weight_size,
            expert_up_scale_size,
            expert_down_scale_size,
            expert_up_bias_size,
            expert_down_bias_size,
            HEADER_SIZE,
            0, 0, 0, 0, 0, 0, 0
        )
        assert len(header) == HEADER_SIZE
        f.write(header)
        
        # Write dummy data for each layer and expert
        pattern = struct.pack('<II', 0xDEADBEEF, 0xCAFEBABE)  # Recognizable pattern
        bytes_written = 0
        
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                # Create layer/expert-specific pattern for verification
                layer_pattern = struct.pack('<II', layer_idx, expert_idx)
                
                # Write up_weight (fill with pattern)
                chunk = (layer_pattern * (expert_up_weight_size // 8 + 1))[:expert_up_weight_size]
                f.write(chunk)
                bytes_written += expert_up_weight_size
                
                # Write up_scale
                chunk = (layer_pattern * (expert_up_scale_size // 8 + 1))[:expert_up_scale_size]
                f.write(chunk)
                bytes_written += expert_up_scale_size
                
                # Write up_bias
                chunk = (layer_pattern * (expert_up_bias_size // 8 + 1))[:expert_up_bias_size]
                f.write(chunk)
                bytes_written += expert_up_bias_size
                
                # Write down_weight
                chunk = (layer_pattern * (expert_down_weight_size // 8 + 1))[:expert_down_weight_size]
                f.write(chunk)
                bytes_written += expert_down_weight_size
                
                # Write down_scale
                chunk = (layer_pattern * (expert_down_scale_size // 8 + 1))[:expert_down_scale_size]
                f.write(chunk)
                bytes_written += expert_down_scale_size
                
                # Write down_bias
                chunk = (layer_pattern * (expert_down_bias_size // 8 + 1))[:expert_down_bias_size]
                f.write(chunk)
                bytes_written += expert_down_bias_size
            
            if (layer_idx + 1) % 4 == 0 or layer_idx == num_layers - 1:
                print(f"  Progress: {layer_idx + 1}/{num_layers} layers, {bytes_written/1024/1024/1024:.2f} GB")
    
    file_size = os.path.getsize(output_path)
    print(f"\nTest OTD file created:")
    print(f"  File size: {file_size/1024/1024/1024:.2f} GB")


def verify_otd_file(file_path: str):
    """Verify an OTD file's header and structure."""
    print(f"\n=== 驗證 OTD 檔案: {os.path.basename(file_path)} ===")
    
    with open(file_path, 'rb') as f:
        # Read header
        header_data = f.read(HEADER_SIZE)
        
        # Parse header
        magic = header_data[0:4]
        version, num_layers, num_experts = struct.unpack_from('<III', header_data, 4)
        (up_weight_size, down_weight_size, up_scale_size, down_scale_size,
         up_bias_size, down_bias_size, data_offset) = struct.unpack_from('<7Q', header_data, 16)
        
        print(f"  Magic: {magic.decode('ascii', errors='ignore')} {'✓' if magic == HEADER_MAGIC else '✗'}")
        print(f"  Version: {version}")
        print(f"  Num Layers: {num_layers}")
        print(f"  Num Experts: {num_experts}")
        print(f"  Up Weight Size: {up_weight_size/1024/1024:.2f} MB")
        print(f"  Down Weight Size: {down_weight_size/1024/1024:.2f} MB")
        print(f"  Up Scale Size: {up_scale_size/1024:.2f} KB")
        print(f"  Down Scale Size: {down_scale_size/1024:.2f} KB")
        print(f"  Up Bias Size: {up_bias_size/1024:.2f} KB")
        print(f"  Down Bias Size: {down_bias_size/1024:.2f} KB")
        print(f"  Data Offset: {data_offset} bytes")
        
        # Calculate expected size
        per_expert = up_weight_size + down_weight_size + up_scale_size + down_scale_size + up_bias_size + down_bias_size
        expected_size = HEADER_SIZE + per_expert * num_experts * num_layers
        
        f.seek(0, 2)
        actual_size = f.tell()
        
        print(f"\n  Per Expert Size: {per_expert/1024/1024:.2f} MB")
        print(f"  Expected File Size: {expected_size/1024/1024/1024:.2f} GB")
        print(f"  Actual File Size: {actual_size/1024/1024/1024:.2f} GB")
        print(f"  Size Match: {'✓' if actual_size == expected_size else '✗ (差異: ' + str(actual_size - expected_size) + ' bytes)'}")


def main():
    parser = argparse.ArgumentParser(description='Extract MoE weights for OTD feature (v2)')
    parser.add_argument('--model', type=str, help='Path to OpenVINO model directory')
    parser.add_argument('--output', type=str, help='Output OTD file path')
    parser.add_argument('--create-test', action='store_true', help='Create test OTD file with simulated data')
    parser.add_argument('--verify', type=str, help='Verify an existing OTD file')
    parser.add_argument('--num-layers', type=int, default=24, help='Number of MoE layers')
    parser.add_argument('--num-experts', type=int, default=32, help='Number of experts per layer')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_otd_file(args.verify)
        return
    
    if args.create_test:
        if not args.output:
            print("Error: --output is required for --create-test")
            return
        create_test_otd_file(args.output, args.num_layers, args.num_experts)
        return
    
    if args.model:
        if not args.output:
            # Default output path
            args.output = str(Path(args.model) / "moe_weights_otd.bin")
        
        extractor = MoEWeightExtractor(args.model)
        
        if not extractor.load_model():
            return
        
        if not extractor.analyze_moe_structure():
            return
        
        if not extractor.extract_weights():
            print("\nFalling back to test file creation...")
            create_test_otd_file(args.output, extractor.num_layers, extractor.num_experts)
            verify_otd_file(args.output)
            return
        
        if not extractor.write_otd_file(args.output):
            return
        
        verify_otd_file(args.output)
        print("\nDone!")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
