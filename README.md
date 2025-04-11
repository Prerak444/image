# Image Processing Application

An interactive web application for image processing built with Streamlit and OpenCV.

## Features

- Various image processing operations:
  - Smoothing filters (Gaussian, Median, Bilateral)
  - Sharpening filters (Laplacian, Unsharp Mask)
  - Edge detection (Canny, Sobel, Laplacian)
  - Color adjustments (Brightness/Contrast, HSV, Color Balance)
- Image histograms and statistics
- Side-by-side comparison
- Download processed images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-processor.git
cd image-processor

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py