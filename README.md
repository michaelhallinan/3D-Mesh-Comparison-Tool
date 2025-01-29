<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template"> 
    <img src="https://github.com/user-attachments/assets/3f86da63-ac81-47a4-9d88-1f660ca30520" alt="Logo">
  </a>

  <h3 align="center">3D Mesh Comparison Tool</h3>

  <p align="center">
    An easy way to grab descriptive statistics and visualize 3D mesh files (.obj/.stl)
    <br />
    <a href="https://github.com/michaelhallinan/3D-Mesh-Comparison-Tool/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/michaelhallinan/3D-Mesh-Comparison-Tool/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
    <br /> 
    <br /> 
    <img src="https://img.shields.io/github/repo-size/michaelhallinan/3D-Mesh-Comparison-Tool" alt="GitHub repo size">
    <img src="https://img.shields.io/github/contributors/michaelhallinan/3D-Mesh-Comparison-Tool" alt="GitHub contributors">
</p>
</div>

## Description
The 3D Mesh Comparison Tool is a Python-based graphical application that allows users to compare two 3D mesh models. It evaluates differences in structure using metrics like Hausdorff Distance and RMSE, provides visualizations, and generates detailed PDF reports of this information. 

## Inputs
- Two 3D mesh models in `.obj` or `.stl` format.
- Optional settings for automatic scaling (Bounding Box, Surface Area, or Volume).
- Enable/disable visualization of differences.

## Outputs
- Hausdorff Distance and RMSE metrics for model comparison.
- A visualization of differences (if enabled).
- A detailed PDF report containing comparison results and images.

## Requirements
Ensure Python 3.11.0 is installed along with the required dependencies:
```sh
pip install numpy scipy trimesh scikit-learn matplotlib customtkinter reportlab pillow
```

## Usage
1. Run the application:
   ```sh
   python Final_3DMesh_Comparison.py
   ```
2. Load two 3D models for comparison.
3. Select scaling options and enable visualization if needed.
4. Click 'Compare Models' to compute differences.
5. Export results as a PDF report if required.

