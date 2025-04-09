#!/usr/bin/env python3
"""
AprilTag Generator Utility

This script helps with generating and saving AprilTag images.
Requires the pupil_apriltags and opencv-python packages.

Usage:
    python generate_apriltags.py --family tag36h11 --ids 0,1,2,3 --size 8 --output ./tags/
"""

import cv2
import numpy as np
import os
import argparse
from pupil_apriltags import tag36h11, tag25h9, tag16h5
import matplotlib.pyplot as plt

# Map family names to tag generating functions
TAG_FAMILIES = {
    'tag36h11': tag36h11,
    'tag25h9': tag25h9,
    'tag16h5': tag16h5
}

def generate_apriltag(family, tag_id, tag_size=8, border=1):
    """
    Generate an AprilTag image.
    
    Args:
        family: String tag family name ('tag36h11', 'tag25h9', 'tag16h5')
        tag_id: Tag ID
        tag_size: Size of the tag as a multiple of the bit size
        border: Width of white border around the tag
    
    Returns:
        numpy.ndarray: Generated tag image
    """
    if family not in TAG_FAMILIES:
        raise ValueError(f"Family {family} not supported. Use one of: {list(TAG_FAMILIES.keys())}")
    
    tag_generator = TAG_FAMILIES[family]
    tag_img = tag_generator(tag_id)
    
    # Add border
    if border > 0:
        h, w = tag_img.shape
        bordered = np.ones((h + 2*border, w + 2*border), dtype=tag_img.dtype)
        bordered[border:-border, border:-border] = tag_img
        tag_img = bordered
    
    # Resize to tag_size
    bit_size = tag_size // (tag_img.shape[0])
    if bit_size < 1:
        bit_size = 1
    
    large_tag = np.zeros((tag_img.shape[0] * bit_size, tag_img.shape[1] * bit_size), dtype=np.uint8)
    for i in range(tag_img.shape[0]):
        for j in range(tag_img.shape[1]):
            large_tag[i*bit_size:(i+1)*bit_size, j*bit_size:(j+1)*bit_size] = tag_img[i, j] * 255
    
    return large_tag

def save_tag(tag_img, output_dir, family, tag_id, dpi=300):
    """Save the tag image to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as PNG
    filename_png = os.path.join(output_dir, f"{family}_{tag_id}.png")
    cv2.imwrite(filename_png, tag_img)
    
    # Also save as PDF for printing
    filename_pdf = os.path.join(output_dir, f"{family}_{tag_id}.pdf")
    plt.figure(figsize=(8.27, 11.69))  # A4 size
    plt.imshow(tag_img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename_pdf, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filename_png, filename_pdf

def main():
    parser = argparse.ArgumentParser(description="Generate AprilTag images")
    parser.add_argument('--family', type=str, default='tag36h11',
                        choices=list(TAG_FAMILIES.keys()),
                        help='AprilTag family')
    parser.add_argument('--ids', type=str, default='0',
                        help='Comma-separated list of tag IDs to generate')
    parser.add_argument('--size', type=int, default=8,
                        help='Size multiplier for the tag')
    parser.add_argument('--border', type=int, default=2,
                        help='Width of white border around the tag')
    parser.add_argument('--output', type=str, default='./tags',
                        help='Output directory for the tag images')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for PDF output')
    args = parser.parse_args()
    
    # Parse tag IDs
    try:
        tag_ids = [int(id_str) for id_str in args.ids.split(',')]
    except ValueError:
        print("Error: Tag IDs must be comma-separated integers")
        return
    
    # Generate and save tags
    for tag_id in tag_ids:
        print(f"Generating {args.family} tag with ID {tag_id}...")
        
        tag_img = generate_apriltag(args.family, tag_id, args.size, args.border)
        
        png_path, pdf_path = save_tag(tag_img, args.output, args.family, tag_id, args.dpi)
        
        print(f"  Saved to {png_path} and {pdf_path}")
    
    print("\nAll tags generated successfully!")
    print(f"The tags are available in the '{args.output}' directory.")
    print("For best results when printing:")
    print("1. Print at 100% scale (no resizing)")
    print("2. Use high quality paper")
    print("3. Ensure there is good lighting when using the tags")

if __name__ == "__main__":
    main()