import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os
from PIL import Image

class ModernMeshComparisonApp:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("3D Mesh Comparison Tool")
        self.root.geometry("1000x800")

        # Initialize variables first
        self.file1_path = None
        self.file2_path = None
        self.comparison_complete = False
        self.results = None
        self.visualization_enabled = tk.BooleanVar(value=True)

        # Create main container with padding
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Create left panel for controls
        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))

        # Create right panel for visualization
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="both", expand=True)

        # Create visualization frame
        self.visualization_frame = ctk.CTkFrame(self.right_panel)
        self.visualization_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create UI elements
        self._create_file_selection()
        self._create_scaling_options()
        self._create_visualization_options()
        self._create_comparison_options()
        self._create_progress_section()

    def _create_file_selection(self):
        # File Selection Section
        file_frame = ctk.CTkFrame(self.left_panel)
        file_frame.pack(fill="x", pady=(0, 10), padx=10)

        ctk.CTkLabel(file_frame, text="File Selection", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        # First file selection
        self.file1_button = ctk.CTkButton(
            file_frame,
            text="Select Reference Model",
            command=self.load_file1
        )
        self.file1_button.pack(pady=5, padx=10, fill="x")

        self.file1_label = ctk.CTkLabel(file_frame, text="No file selected", wraplength=250)
        self.file1_label.pack(pady=(0, 10))

        # Second file selection
        self.file2_button = ctk.CTkButton(
            file_frame,
            text="Select Comparison Model",
            command=self.load_file2
        )
        self.file2_button.pack(pady=5, padx=10, fill="x")

        self.file2_label = ctk.CTkLabel(file_frame, text="No file selected", wraplength=250)
        self.file2_label.pack(pady=(0, 10))

    def _create_scaling_options(self):
        # Scaling Options Section
        scaling_frame = ctk.CTkFrame(self.left_panel)
        scaling_frame.pack(fill="x", pady=(0, 10), padx=10)

        ctk.CTkLabel(scaling_frame, text="Scaling Options", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        self.auto_scale_var = tk.BooleanVar(value=True)
        self.auto_scale_cb = ctk.CTkCheckBox(
            scaling_frame,
            text="Enable automatic scaling",
            variable=self.auto_scale_var,
            command=self._create_scaling_options
        )
        self.auto_scale_cb.pack(pady=5)

        # Scaling method selection
        self.scaling_method = tk.StringVar(value="bbox")

        methods_frame = ctk.CTkFrame(scaling_frame)
        methods_frame.pack(fill="x", pady=5, padx=10)

        for text, value in [("Bounding Box", "bbox"),
                            ("Surface Area", "area"),
                            ("Volume", "volume")]:
            ctk.CTkRadioButton(
                methods_frame,
                text=text,
                variable=self.scaling_method,
                value=value
            ).pack(pady=2)

    def _create_comparison_options(self):
        # Comparison Options Section
        comparison_frame = ctk.CTkFrame(self.left_panel)
        comparison_frame.pack(fill="x", pady=(0, 10), padx=10)

        ctk.CTkLabel(comparison_frame, text="Actions", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        self.compare_button = ctk.CTkButton(
            comparison_frame,
            text="Compare Models",
            command=self.compare_files,
            state="disabled"
        )
        self.compare_button.pack(pady=5, padx=10, fill="x")

        self.export_button = ctk.CTkButton(
            comparison_frame,
            text="Export PDF Report",
            command=self.export_pdf_report,
            state="disabled"
        )
        self.export_button.pack(pady=5, padx=10, fill="x")

    def _create_progress_section(self):
        # Progress Section
        progress_frame = ctk.CTkFrame(self.left_panel)
        progress_frame.pack(fill="x", pady=(0, 10), padx=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(pady=10, padx=10, fill="x")
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.status_label.pack(pady=(0, 10))

    def _create_visualization_panel(self):
        # Clear existing content
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        # Add initial message
        ctk.CTkLabel(
            self.visualization_frame,
            text="Visualization will appear here after comparison",
            font=("Helvetica", 14)
        ).pack(pady=20)

    def _create_visualization_options(self):
        # Visualization Options Section
        viz_frame = ctk.CTkFrame(self.left_panel)
        viz_frame.pack(fill="x", pady=(0, 10), padx=10)

        ctk.CTkLabel(viz_frame, text="Visualization Options", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        # Visualization toggle
        self.viz_toggle = ctk.CTkCheckBox(
            viz_frame,
            text="Enable visualization",
            variable=self.visualization_enabled
        )
        self.viz_toggle.pack(pady=5)

        # Warning label
        warning_text = ("Note: Visualization may not work correctly with models that have "
                        "significantly different vertex counts or large geometric differences. "
                        "For best results, use models with similar complexity.")

        warning_label = ctk.CTkLabel(
            viz_frame,
            text=warning_text,
            wraplength=225,
            text_color="red"
        )
        warning_label.pack(pady=5)


    def load_file1(self):
        self.file1_path = filedialog.askopenfilename(
            filetypes=[("3D Model Files", "*.obj *.stl")]
        )
        if self.file1_path:
            self.file1_label.configure(text=os.path.basename(self.file1_path))
            self.check_files_loaded()

    def load_file2(self):
        self.file2_path = filedialog.askopenfilename(
            filetypes=[("3D Model Files", "*.obj *.stl")]
        )
        if self.file2_path:
            self.file2_label.configure(text=os.path.basename(self.file2_path))
            self.check_files_loaded()

    def check_files_loaded(self):
        if self.file1_path and self.file2_path:
            self.compare_button.configure(state="normal")

    def update_progress(self, value, status):
        self.progress_bar.set(value / 100)
        self.status_label.configure(text=status)
        self.root.update_idletasks()

    def create_comparison_visualization(self, mesh1: object, mesh2: object) -> object:
        self.update_progress(70, "Creating visualization...")

        # Clear previous visualization
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        # Create new figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot reference mesh
        ax.scatter(mesh1.vertices[:, 0], mesh1.vertices[:, 1], mesh1.vertices[:, 2],
                   c='blue', alpha=0.5, label='Reference Model', s=1)

        # Plot aligned comparison mesh
        ax.scatter(mesh2.vertices[:, 0], mesh2.vertices[:, 1], mesh2.vertices[:, 2],
                   c='red', alpha=0.5, label='Comparison Model (Aligned)', s=1)

        # Calculate and plot differences
        kdtree = cKDTree(mesh2.vertices)
        distances, _ = kdtree.query(mesh1.vertices)
        threshold = np.mean(distances) + np.std(distances)
        diff_points = mesh1.vertices[distances > threshold]

        if len(diff_points) > 0:
            ax.scatter(diff_points[:, 0], diff_points[:, 1], diff_points[:, 2],
                       c='green', alpha=1, label='Major Differences', s=2)

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Auto-scale axes
        max_range = np.array([
            ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
            ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
            ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ]).max() / 2.0

        mid_x = (ax.get_xlim3d()[1] + ax.get_xlim3d()[0]) / 2.0
        mid_y = (ax.get_ylim3d()[1] + ax.get_ylim3d()[0]) / 2.0
        mid_z = (ax.get_zlim3d()[1] + ax.get_zlim3d()[0]) / 2.0

        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Save figure and convert to PIL Image
        plt.savefig('temp_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Import PIL and create image
        from PIL import Image
        pil_image = Image.open('temp_visualization.png')

        # Create CTkImage
        img = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(600, 600)
        )

        # Create and pack the label
        label = ctk.CTkLabel(self.visualization_frame, image=img, text="")
        label.image = img  # Keep a reference
        label.pack(fill="both", expand=True)

    def export_pdf_report(self):
        if not self.comparison_complete or not self.results:
            messagebox.showerror("Error", "Please complete a comparison first.")
            return

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="3D_Model_Comparison_Report.pdf"
        )

        if not file_path:
            return

        # Create PDF
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            spaceAfter=30
        )
        story.append(Paragraph("3D Model Comparison Report", title_style))
        story.append(Spacer(1, 20))

        # Add file information
        story.append(Paragraph(f"Reference Model: {os.path.basename(self.file1_path)}", styles['Heading2']))
        story.append(Paragraph(f"Comparison Model: {os.path.basename(self.file2_path)}", styles['Heading2']))
        story.append(Spacer(1, 20))

        # Add scaling information
        story.append(Paragraph("Scaling Settings", styles['Heading2']))
        story.append(
            Paragraph(f"Auto-scaling: {'Enabled' if self.auto_scale_var.get() else 'Disabled'}", styles['Normal']))
        story.append(Paragraph(f"Scaling Method: {self.scaling_method.get() if self.auto_scale_var.get() else 'N/A'}",
                               styles['Normal']))
        story.append(Spacer(1, 20))

        # Add metrics
        story.append(Paragraph("Comparison Metrics", styles['Heading2']))
        metrics_text = f"""
        Hausdorff Distance: {self.results['hausdorff']:.4f}<br/>
        RMSE: {self.results['rmse']:.4f}<br/>
        Surface Area (Reference): {self.results['area1']:.4f}<br/>
        Surface Area (Comparison): {self.results['area2']:.4f}<br/>
        Roughness (Reference): {self.results['roughness1']:.4f}<br/>
        Roughness (Comparison): {self.results['roughness2']:.4f}<br/>
        Vertices (Reference): {self.results['vertices1']}<br/>
        Vertices (Comparison): {self.results['vertices2']}
        """
        story.append(Paragraph(metrics_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Add visualization only if enabled and file exists
        if self.visualization_enabled.get() and os.path.exists('temp_visualization.png'):
            story.append(Paragraph("Visualization", styles['Heading2']))
            story.append(Image('temp_visualization.png', width=400, height=400))

        # Build PDF
        doc.build(story)

        # Clean up temporary files
        if os.path.exists('temp_visualization.png'):
            os.remove('temp_visualization.png')

        messagebox.showinfo("Success", f"Report saved to {file_path}")

    def align_meshes(self, source_mesh, target_mesh):
        """
        Align source mesh to target mesh using ICP (Iterative Closest Point).
        Returns the aligned source mesh.
        """
        self.update_progress(35, "Aligning meshes using ICP...")

        # Convert to point clouds for alignment
        source_points = source_mesh.vertices
        target_points = target_mesh.vertices

        # Center both point clouds
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)

        source_centered = source_points - source_center
        target_centered = target_points - target_center

        # Initialize transformation
        R = np.eye(3)
        t = np.zeros(3)

        # ICP parameters
        max_iterations = 50
        convergence_threshold = 5
        prev_error = float('inf')

        for iteration in range(max_iterations):
            # Find nearest neighbors
            tree = cKDTree(target_centered)
            distances, indices = tree.query(source_centered)

            corresponding_points = target_centered[indices]

            # Compute centroids
            source_centroid = np.mean(source_centered, axis=0)
            target_centroid = np.mean(corresponding_points, axis=0)

            # Center the point sets
            source_centered_centroid = source_centered - source_centroid
            target_centered_centroid = corresponding_points - target_centroid

            # Compute optimal rotation
            H = np.dot(source_centered_centroid.T, target_centered_centroid)
            U, S, Vt = np.linalg.svd(H)
            R_opt = np.dot(Vt.T, U.T)

            # Ensure right-handed coordinate system
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = np.dot(Vt.T, U.T)

            # Compute optimal translation
            t_opt = target_centroid - np.dot(source_centroid, R_opt.T)

            # Update transformation
            R = np.dot(R_opt, R)
            t = np.dot(t_opt, R) + t

            # Apply transformation
            source_centered = np.dot(source_points - source_center, R_opt.T) + t_opt

            # Check convergence
            current_error = np.mean(distances)
            if abs(prev_error - current_error) < convergence_threshold:
                break
            prev_error = current_error

            self.update_progress(35 + int(10 * iteration / max_iterations),
                                 f"ICP iteration {iteration + 1}/{max_iterations}")

        # Create aligned mesh
        aligned_mesh = source_mesh.copy()
        aligned_mesh.vertices = np.dot(source_mesh.vertices - source_center, R.T) + target_center + t

        return aligned_mesh

    def compare_files(self):
        self.compare_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.update_progress(0, "Starting comparison...")

        try:
            # Load models
            self.update_progress(20, "Loading reference model...")
            mesh1 = trimesh.load(self.file1_path)
            self.update_progress(30, "Loading comparison model...")
            mesh2 = trimesh.load(self.file2_path)

            # Store original vertex counts before any modifications
            original_vertices1 = len(mesh1.vertices)
            original_vertices2 = len(mesh2.vertices)

            # Check for significant vertex count difference
            vertex_ratio = max(original_vertices1, original_vertices2) / min(original_vertices1, original_vertices2)
            if vertex_ratio > 2 and self.visualization_enabled.get():
                if not messagebox.askyesno("Warning",
                                           "The models have significantly different vertex counts "
                                           f"(ratio {vertex_ratio:.1f}:1). This may result in poor visualization. "
                                           "Continue with visualization anyway?"):
                    self.visualization_enabled.set(False)

            # Scale if needed
            if self.auto_scale_var.get():
                self.update_progress(40, f"Scaling models using {self.scaling_method.get()} method...")
                if self.scaling_method.get() == 'bbox':
                    size1 = np.linalg.norm(np.diagonal(mesh1.bounding_box.bounds))
                    size2 = np.linalg.norm(np.diagonal(mesh2.bounding_box.bounds))
                elif self.scaling_method.get() == 'area':
                    size1 = mesh1.area
                    size2 = mesh2.area
                else:  # volume
                    size1 = mesh1.volume
                    size2 = mesh2.volume

                if size1 > size2:
                    mesh2 = self.scale_mesh(mesh2, mesh1, self.scaling_method.get())
                else:
                    mesh1 = self.scale_mesh(mesh1, mesh2, self.scaling_method.get())

            # Align meshes using ICP
            mesh2_aligned = self.align_meshes(mesh2, mesh1)

            # Calculate metrics
            self.update_progress(50, "Calculating metrics...")
            hausdorff_dist = self.hausdorff_distance(mesh1, mesh2_aligned)
            rmse = self.calculate_rmse(mesh1, mesh2_aligned)

            # Store results
            self.results = {
                'hausdorff': hausdorff_dist,
                'rmse': rmse,
                'area1': mesh1.area,
                'area2': mesh2_aligned.area,
                'roughness1': self.calculate_roughness(mesh1),
                'roughness2': self.calculate_roughness(mesh2_aligned),
                'vertices1': original_vertices1,
                'vertices2': original_vertices2
            }

            # Create visualization only if enabled
            if self.visualization_enabled.get():
                self.create_comparison_visualization(mesh1, mesh2_aligned)
            else:
                # Clear visualization panel
                for widget in self.visualization_frame.winfo_children():
                    widget.destroy()
                ctk.CTkLabel(
                    self.visualization_frame,
                    text="Visualization disabled",
                    font=("Helvetica", 14)
                ).pack(pady=20)

            self.update_progress(100, "Comparison complete!")
            self.comparison_complete = True
            self.export_button.configure(state="normal")

            # Display results in the interface
            self.show_results()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.update_progress(0, "Error occurred during comparison")

        finally:
            self.compare_button.configure(state="normal")


    def show_results(self):
        # Create results window
        results_window = ctk.CTkToplevel(self.root)
        results_window.title("Comparison Results")
        results_window.geometry("400x500")

        # Results container
        results_frame = ctk.CTkFrame(results_window)
        results_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            results_frame,
            text="Comparison Results",
            font=("Helvetica", 18, "bold")
        ).pack(pady=(0, 20))

        metrics = [
            ("Hausdorff Distance", f"{self.results['hausdorff']:.4f}"),
            ("RMSE", f"{self.results['rmse']:.4f}"),
            ("Reference Surface Area", f"{self.results['area1']:.4f}"),
            ("Comparison Surface Area", f"{self.results['area2']:.4f}"),
            ("Reference Roughness", f"{self.results['roughness1']:.4f}"),
            ("Comparison Roughness", f"{self.results['roughness2']:.4f}"),
            ("Reference Vertices", f"{self.results['vertices1']}"),  # New line
            ("Comparison Vertices", f"{self.results['vertices2']}")  # New line
        ]

        for label, value in metrics:
            metric_frame = ctk.CTkFrame(results_frame)
            metric_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(
                metric_frame,
                text=label,
                font=("Helvetica", 12, "bold")
            ).pack(side="left", padx=10)

            ctk.CTkLabel(
                metric_frame,
                text=value
            ).pack(side="right", padx=10)


    def scale_mesh(self, mesh_to_scale, reference_mesh, method='bbox'):
        """Scale mesh_to_scale to match the size of reference_mesh using the specified method."""
        if method == 'bbox':
            # Scale based on bounding box diagonal
            ref_bbox = np.diagonal(reference_mesh.bounding_box.bounds)
            scale_bbox = np.diagonal(mesh_to_scale.bounding_box.bounds)
            ref_size = np.linalg.norm(ref_bbox)
            scale_size = np.linalg.norm(scale_bbox)
            scale_factor = ref_size / scale_size
        elif method == 'area':
            # Scale based on surface area
            ref_area = reference_mesh.area
            scale_area = mesh_to_scale.area
            scale_factor = np.sqrt(ref_area / scale_area)
        else:  # method == 'volume'
            # Scale based on volume
            ref_volume = reference_mesh.volume
            scale_volume = mesh_to_scale.volume
            scale_factor = np.cbrt(ref_volume / scale_volume)

        # Create a copy of the mesh and scale it
        scaled_mesh = mesh_to_scale.copy()
        scaled_mesh.apply_scale(scale_factor)

        return scaled_mesh


    def hausdorff_distance(self, mesh1, mesh2):
        """Calculate the Hausdorff distance between two meshes."""
        points1, points2 = mesh1.vertices, mesh2.vertices
        kdtree1 = cKDTree(points1)
        kdtree2 = cKDTree(points2)

        distances1, _ = kdtree2.query(points1)
        distances2, _ = kdtree1.query(points2)

        return max(distances1.max(), distances2.max())


    def calculate_rmse(self, mesh1, mesh2):
        """Calculate the Root Mean Square Error between two meshes."""
        points1, points2 = mesh1.vertices, mesh2.vertices
        if points1.shape[0] > points2.shape[0]:
            points2 = np.pad(points2, [(0, points1.shape[0] - points2.shape[0]), (0, 0)], mode='constant')
        elif points1.shape[0] < points2.shape[0]:
            points1 = np.pad(points1, [(0, points2.shape[0] - points1.shape[0]), (0, 0)], mode='constant')

        return np.sqrt(mean_squared_error(points1, points2))


    def calculate_roughness(self, mesh):
        """Calculate the arithmetic average roughness of a mesh."""
        z_coords = mesh.vertices[:, 2]
        average_height = np.mean(z_coords)
        return np.mean(np.abs(z_coords - average_height))



if __name__ == "__main__":
    app = ModernMeshComparisonApp()
    app.root.mainloop()
