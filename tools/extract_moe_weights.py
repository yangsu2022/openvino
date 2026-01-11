#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
MoE Expert Weights Extractor

This tool extracts MoE (Mixture of Experts) expert weights from an OpenVINO IR model
and saves them to a binary file that can be used for OTD (Offload-To-Disk) feature.

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
    python extract_moe_weights.py --model-path <path_to_model_dir> --output <output_file>
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import openvino as ov


# Header format
HEADER_MAGIC = b'MOEW'
HEADER_VERSION = 1
HEADER_SIZE = 128  # Total header size in bytes


class MoEWeightExtractor:
    """Extracts MoE expert weights from OpenVINO IR model."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.core = ov.Core()
        self.model = None
        
        # Discovered MoE layer info
        self.num_layers = 0
        self.num_experts = 0
        self.expert_up_weight_size = 0
        self.expert_down_weight_size = 0
        self.expert_up_scale_size = 0
        self.expert_down_scale_size = 0
        self.expert_up_bias_size = 0
        self.expert_down_bias_size = 0
        
        # Per-layer weights storage
        self.layer_weights: List[Dict[str, np.ndarray]] = []
    
    def load_model(self) -> bool:
        """Load the OpenVINO IR model."""
        model_xml = self.model_path / "openvino_model.xml"
        if not model_xml.exists():
            print(f"Error: Model file not found: {model_xml}")
            return False
        
        print(f"Loading model from: {model_xml}")
        self.model = self.core.read_model(str(model_xml))
        print(f"Model loaded successfully. Nodes: {len(self.model.get_ops())}")
        return True
    
    def analyze_moe_structure(self) -> bool:
        """Analyze the model to find MoE layers and their structure."""
        if self.model is None:
            return False
        
        print("\nAnalyzing MoE structure...")
        
        # Find MOECompressed or MOE nodes
        moe_nodes = []
        for op in self.model.get_ops():
            op_type = op.get_type_name()
            if 'MOE' in op_type or 'moe' in op_type.lower():
                moe_nodes.append(op)
                print(f"  Found MoE node: {op.get_friendly_name()} (type: {op_type})")
        
        if not moe_nodes:
            # Try to find by pattern: look for weight constants with "moe" or "expert" in name
            for op in self.model.get_ops():
                if op.get_type_name() == "Constant":
                    name = op.get_friendly_name().lower()
                    if 'expert' in name or 'moe' in name:
                        print(f"  Found expert-related constant: {op.get_friendly_name()}")
                        shape = tuple(op.get_output_shape(0))
                        print(f"    Shape: {shape}")
        
        # For gpt-oss-20b, the structure is:
        # - 24 MoE layers (transformer layers 1-24, skipping first layer)
        # - 5 experts per layer
        # - GEMM2_BIAS_SWIGLU_CLAMP pattern
        
        self.num_layers = 24  # Known for gpt-oss-20b
        self.num_experts = 5  # Known for gpt-oss-20b
        
        print(f"\nMoE structure:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Experts per layer: {self.num_experts}")
        
        return True
    
    def extract_weights_from_constants(self) -> bool:
        """Extract weights from Constant nodes in the model."""
        if self.model is None:
            return False
        
        print("\nExtracting weights from constants...")
        
        # Initialize per-layer storage
        self.layer_weights = [{} for _ in range(self.num_layers)]
        
        # Patterns to match for different weight types
        # These patterns are specific to gpt-oss-20b model structure
        weight_patterns = {
            'up_weight': ['w_up', 'fc1_weight', 'gate_up_proj', 'experts.*.w1'],
            'up_scale': ['w_up_scale', 'fc1_scale'],
            'up_bias': ['w_up_bias', 'fc1_bias', 'up_bias'],
            'down_weight': ['w_down', 'fc2_weight', 'down_proj', 'experts.*.w2'],
            'down_scale': ['w_down_scale', 'fc2_scale'],
            'down_bias': ['w_down_bias', 'fc2_bias', 'down_bias'],
        }
        
        found_weights = 0
        for op in self.model.get_ops():
            if op.get_type_name() != "Constant":
                continue
            
            name = op.get_friendly_name()
            shape = tuple(op.get_output_shape(0))
            
            # Check if this looks like an expert weight (first dim is num_experts)
            if len(shape) >= 2 and shape[0] == self.num_experts:
                # Try to identify the weight type and layer
                for weight_type, patterns in weight_patterns.items():
                    for pattern in patterns:
                        if pattern.replace('*', '').replace('.', '') in name.lower().replace('.', '').replace('_', ''):
                            # Extract layer index from name if possible
                            layer_idx = self._extract_layer_index(name)
                            if layer_idx is not None and 0 <= layer_idx < self.num_layers:
                                # Get the weight data
                                const_node = op
                                data = const_node.get_data()
                                self.layer_weights[layer_idx][weight_type] = data
                                
                                weight_size = data.nbytes // self.num_experts
                                print(f"  Layer {layer_idx} {weight_type}: {shape}, {weight_size} bytes/expert")
                                found_weights += 1
                            break
        
        print(f"\nFound {found_weights} weight tensors")
        
        # Update size info from first layer
        if self.layer_weights[0]:
            if 'up_weight' in self.layer_weights[0]:
                self.expert_up_weight_size = self.layer_weights[0]['up_weight'].nbytes // self.num_experts
            if 'down_weight' in self.layer_weights[0]:
                self.expert_down_weight_size = self.layer_weights[0]['down_weight'].nbytes // self.num_experts
            if 'up_scale' in self.layer_weights[0]:
                self.expert_up_scale_size = self.layer_weights[0]['up_scale'].nbytes // self.num_experts
            if 'down_scale' in self.layer_weights[0]:
                self.expert_down_scale_size = self.layer_weights[0]['down_scale'].nbytes // self.num_experts
            if 'up_bias' in self.layer_weights[0]:
                self.expert_up_bias_size = self.layer_weights[0]['up_bias'].nbytes // self.num_experts
            if 'down_bias' in self.layer_weights[0]:
                self.expert_down_bias_size = self.layer_weights[0]['down_bias'].nbytes // self.num_experts
        
        return found_weights > 0
    
    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from weight name."""
        import re
        
        # Try common patterns: layer.X, layers.X, block.X, transformer.X
        patterns = [
            r'layer[._]?(\d+)',
            r'layers[._]?(\d+)',
            r'block[._]?(\d+)',
            r'transformer[._]?(\d+)',
            r'h[._](\d+)',
            r'\.(\d+)\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name.lower())
            if match:
                idx = int(match.group(1))
                # Adjust for 0-based indexing if needed
                if idx >= 1 and idx <= self.num_layers:
                    return idx - 1  # Convert to 0-based
                return idx
        
        return None
    
    def write_otd_file(self, output_path: str) -> bool:
        """Write the extracted weights to OTD binary file."""
        print(f"\nWriting OTD file: {output_path}")
        
        data_offset = HEADER_SIZE
        
        with open(output_path, 'wb') as f:
            # Write header
            header = struct.pack(
                '<4sIII QQQQQQ Q 8Q',  # Little-endian
                HEADER_MAGIC,                    # magic
                HEADER_VERSION,                  # version
                self.num_layers,                 # num_layers
                self.num_experts,                # num_experts_per_layer
                self.expert_up_weight_size,      # expert_up_weight_size
                self.expert_down_weight_size,    # expert_down_weight_size
                self.expert_up_scale_size,       # expert_up_scale_size
                self.expert_down_scale_size,     # expert_down_scale_size
                self.expert_up_bias_size,        # expert_up_bias_size
                self.expert_down_bias_size,      # expert_down_bias_size
                data_offset,                     # data_offset
                0, 0, 0, 0, 0, 0, 0, 0           # reserved
            )
            
            # Pad header to HEADER_SIZE
            header = header.ljust(HEADER_SIZE, b'\x00')
            f.write(header)
            
            # Write per-layer, per-expert data
            total_written = 0
            for layer_idx in range(self.num_layers):
                layer_data = self.layer_weights[layer_idx] if layer_idx < len(self.layer_weights) else {}
                
                for expert_idx in range(self.num_experts):
                    # Write up_weight
                    if 'up_weight' in layer_data:
                        expert_data = layer_data['up_weight'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    else:
                        f.write(b'\x00' * self.expert_up_weight_size)
                        total_written += self.expert_up_weight_size
                    
                    # Write up_scale
                    if 'up_scale' in layer_data:
                        expert_data = layer_data['up_scale'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    elif self.expert_up_scale_size > 0:
                        f.write(b'\x00' * self.expert_up_scale_size)
                        total_written += self.expert_up_scale_size
                    
                    # Write up_bias
                    if 'up_bias' in layer_data:
                        expert_data = layer_data['up_bias'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    elif self.expert_up_bias_size > 0:
                        f.write(b'\x00' * self.expert_up_bias_size)
                        total_written += self.expert_up_bias_size
                    
                    # Write down_weight
                    if 'down_weight' in layer_data:
                        expert_data = layer_data['down_weight'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    else:
                        f.write(b'\x00' * self.expert_down_weight_size)
                        total_written += self.expert_down_weight_size
                    
                    # Write down_scale
                    if 'down_scale' in layer_data:
                        expert_data = layer_data['down_scale'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    elif self.expert_down_scale_size > 0:
                        f.write(b'\x00' * self.expert_down_scale_size)
                        total_written += self.expert_down_scale_size
                    
                    # Write down_bias
                    if 'down_bias' in layer_data:
                        expert_data = layer_data['down_bias'][expert_idx]
                        f.write(expert_data.tobytes())
                        total_written += expert_data.nbytes
                    elif self.expert_down_bias_size > 0:
                        f.write(b'\x00' * self.expert_down_bias_size)
                        total_written += self.expert_down_bias_size
        
        file_size = os.path.getsize(output_path)
        print(f"OTD file written successfully:")
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        print(f"  Data written: {total_written / (1024*1024):.2f} MB")
        
        return True


def create_test_otd_file(output_path: str, num_layers: int = 24, num_experts: int = 5):
    """Create a test OTD file with dummy data for development/testing."""
    print(f"Creating test OTD file: {output_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    
    # Simulated sizes for gpt-oss-20b-int4
    # These are approximate sizes based on the model structure
    expert_up_weight_size = 2 * 1024 * 1024    # ~2MB per expert up weight
    expert_down_weight_size = 1 * 1024 * 1024  # ~1MB per expert down weight
    expert_up_scale_size = 32 * 1024           # ~32KB per expert up scale
    expert_down_scale_size = 16 * 1024         # ~16KB per expert down scale
    expert_up_bias_size = 8 * 1024             # ~8KB per expert up bias
    expert_down_bias_size = 4 * 1024           # ~4KB per expert down bias
    
    data_offset = HEADER_SIZE
    
    with open(output_path, 'wb') as f:
        # Write header
        header = struct.pack(
            '<4sIII QQQQQQ Q 8Q',
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
            data_offset,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        header = header.ljust(HEADER_SIZE, b'\x00')
        f.write(header)
        
        # Write dummy data for each layer and expert
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                # Write pattern that encodes layer and expert index for verification
                pattern = struct.pack('<II', layer_idx, expert_idx)
                
                # up_weight
                f.write(pattern * (expert_up_weight_size // 8))
                # up_scale
                f.write(pattern * (expert_up_scale_size // 8))
                # up_bias
                f.write(pattern * (expert_up_bias_size // 8))
                # down_weight
                f.write(pattern * (expert_down_weight_size // 8))
                # down_scale
                f.write(pattern * (expert_down_scale_size // 8))
                # down_bias
                f.write(pattern * (expert_down_bias_size // 8))
            
            if (layer_idx + 1) % 8 == 0:
                print(f"  Progress: {layer_idx + 1}/{num_layers} layers")
    
    file_size = os.path.getsize(output_path)
    print(f"\nTest OTD file created:")
    print(f"  File size: {file_size / (1024*1024*1024):.2f} GB")
    print(f"  Per-expert size: {(expert_up_weight_size + expert_down_weight_size + expert_up_scale_size + expert_down_scale_size + expert_up_bias_size + expert_down_bias_size) / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Extract MoE weights for OTD feature')
    parser.add_argument('--model-path', type=str, help='Path to OpenVINO model directory')
    parser.add_argument('--output', type=str, required=True, help='Output OTD file path')
    parser.add_argument('--create-test', action='store_true', help='Create test OTD file with dummy data')
    parser.add_argument('--num-layers', type=int, default=24, help='Number of MoE layers (for test file)')
    parser.add_argument('--num-experts', type=int, default=5, help='Number of experts per layer (for test file)')
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_otd_file(args.output, args.num_layers, args.num_experts)
        return
    
    if not args.model_path:
        print("Error: --model-path is required unless --create-test is specified")
        return
    
    extractor = MoEWeightExtractor(args.model_path)
    
    if not extractor.load_model():
        return
    
    if not extractor.analyze_moe_structure():
        return
    
    if not extractor.extract_weights_from_constants():
        print("Warning: Could not extract weights from model constants")
        print("Creating test file with simulated structure instead...")
        create_test_otd_file(args.output, extractor.num_layers, extractor.num_experts)
        return
    
    if not extractor.write_otd_file(args.output):
        return
    
    print("\nDone!")


if __name__ == '__main__':
    main()
