#!/usr/bin/env python
import os
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Prepare car damage report for deployment")
    parser.add_argument('--output_dir', type=str, default="deployment",
                        help="Output directory for the deployment files")
    args = parser.parse_args()

    # Get the current directory
    current_dir = Path.cwd()
    
    # Create deployment directory
    deploy_dir = Path(args.output_dir)
    if deploy_dir.exists():
        print(f"Warning: Output directory {deploy_dir} already exists. Files may be overwritten.")
    else:
        deploy_dir.mkdir(parents=True)
    
    # Create the output directory structure
    output_dir = deploy_dir / "output" / "car_damage_detection"
    uploads_dir = output_dir / "uploads"
    results_dir = output_dir / "results"
    
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy HTML file and rename to index.html
    html_file = current_dir / "car_damage_report.html"
    if html_file.exists():
        shutil.copy(html_file, deploy_dir / "index.html")
        print(f"Copied {html_file} to {deploy_dir / 'index.html'}")
    else:
        print(f"Error: {html_file} not found!")
        return
    
    # Copy uploads and results
    src_uploads = current_dir / "output" / "car_damage_detection" / "uploads"
    src_results = current_dir / "output" / "car_damage_detection" / "results"
    
    if src_uploads.exists():
        for img_file in src_uploads.glob("*.jpg"):
            shutil.copy(img_file, uploads_dir / img_file.name)
            print(f"Copied {img_file.name} to uploads directory")
    else:
        print(f"Warning: Source uploads directory {src_uploads} not found!")
    
    if src_results.exists():
        for img_file in src_results.glob("*.jpg"):
            shutil.copy(img_file, results_dir / img_file.name)
            print(f"Copied {img_file.name} to results directory")
    else:
        print(f"Warning: Source results directory {src_results} not found!")
    
    # Copy deployment instructions
    instructions_file = current_dir / "deploy_instructions.md"
    if instructions_file.exists():
        shutil.copy(instructions_file, deploy_dir / "INSTRUCTIONS.md")
        print(f"Copied deployment instructions to {deploy_dir / 'INSTRUCTIONS.md'}")
    
    print(f"\nDeployment files prepared in: {deploy_dir}")
    print(f"Follow the instructions in {deploy_dir / 'INSTRUCTIONS.md'} to deploy your report.")

if __name__ == "__main__":
    main() 