import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure page
st.set_page_config(page_title="Image Processing", layout="wide", page_icon="🔬")
st.title("Image Processing")

# Custom CSS for modern UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(45deg, #1a1a1a, #2a2a2a) !important;
        color: white !important;
    }
    .comparison-container {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Image upload with drag & drop zone
with st.sidebar.expander("📤 UPLOAD IMAGE", expanded=True):
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Function to display image statistics
def display_stats(image):
    if len(image.shape) == 3:
        mean_val = np.mean(image, axis=(0, 1))
        std_val = np.std(image, axis=(0, 1))
        min_val = np.min(image, axis=(0, 1))
        max_val = np.max(image, axis=(0, 1))
        
        stats = {
            "Mean": f"R: {mean_val[0]:.2f}, G: {mean_val[1]:.2f}, B: {mean_val[2]:.2f}",
            "Std Dev": f"R: {std_val[0]:.2f}, G: {std_val[1]:.2f}, B: {std_val[2]:.2f}",
            "Min": f"R: {min_val[0]}, G: {min_val[1]}, B: {min_val[2]}",
            "Max": f"R: {max_val[0]}, G: {max_val[1]}, B: {max_val[2]}"
        }
    else:
        stats = {
            "Mean": f"{np.mean(image):.2f}",
            "Std Dev": f"{np.std(image):.2f}",
            "Min": f"{np.min(image)}",
            "Max": f"{np.max(image)}"
        }
    
    return stats

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    
    # Main columns for layout
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_column_width=True)
            
        with st.expander("Image Statistics", expanded=False):
            stats = display_stats(image)
            stat_cols = st.columns(4)
            for i, (label, value) in enumerate(stats.items()):
                with stat_cols[i]:
                    st.metric(label, value)
    
    # Processing controls
    with st.sidebar:
        st.markdown("## 🎚 Processing Controls")
        processor = st.radio("Select Operation:", [
            "Smoothing Filters",
            "Sharpening Filters",
            "Edge Detection",
            "Color Adjustments",
            "Transformations",
            "Special Effects"
        ])
        
        # Add download option for processed image
        st.markdown("## 💾 Save Result")
        show_comparison = st.checkbox("Show Side-by-Side Comparison", value=True)
    
    # Processing pipeline
    processed = image.copy()
    with col2:
        st.markdown("### Processed Image")
        
        if processor == "Smoothing Filters":
            smooth_type = st.radio("Filter Type:", ["Gaussian", "Median", "Bilateral", "Box"])
            kernel_size = st.slider("Kernel Size", 3, 25, 9, 2)
            
            if smooth_type == "Gaussian":
                sigma = st.slider("Sigma", 0.1, 5.0, 1.5)
                processed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            elif smooth_type == "Median":
                processed = cv2.medianBlur(image, kernel_size)
            elif smooth_type == "Bilateral":
                d = st.slider("Diameter", 1, 15, 9)
                sigma_color = st.slider("Color Sigma", 1, 200, 75)
                sigma_space = st.slider("Spatial Sigma", 1, 200, 75)
                processed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            else:  # Box filter
                processed = cv2.boxFilter(image, -1, (kernel_size, kernel_size))
        
        elif processor == "Sharpening Filters":
            sharp_type = st.radio("Technique:", ["Laplacian", "Unsharp Mask", "High Pass"])
            
            if sharp_type == "Laplacian":
                strength = st.slider("Strength", 1, 10, 5) 
                kernel = np.array([[0, -1, 0], [-1, strength, -1], [0, -1, 0]])
                processed = cv2.filter2D(image, -1, kernel)
            elif sharp_type == "Unsharp Mask":
                strength = st.slider("Strength", 0.5, 3.0, 1.5)
                blur = cv2.GaussianBlur(image, (0,0), 3)
                processed = cv2.addWeighted(image, strength, blur, -0.5, 0)
            else:  # High Pass
                kernel_size = st.slider("Kernel Size", 3, 15, 9, 2)
                blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                processed = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
        
        elif processor == "Edge Detection":
            edge_type = st.selectbox("Detection Method:", ["Canny", "Sobel", "Laplacian", "Prewitt", "Roberts"])
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if edge_type == "Canny":
                threshold1 = st.slider("Low Threshold", 0, 255, 50)
                threshold2 = st.slider("High Threshold", 0, 255, 150)
                processed = cv2.Canny(gray, threshold1, threshold2)
                # Convert back to 3 channels for consistent display
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif edge_type == "Sobel":
                dx = st.slider("X Derivative", 0, 2, 1)
                dy = st.slider("Y Derivative", 0, 2, 1)
                ksize = st.slider("Kernel Size", 1, 7, 5, 2)
                sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
                processed = np.uint8(np.absolute(sobel))
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif edge_type == "Laplacian":
                ksize = st.slider("Kernel Size", 1, 7, 3, 2)
                processed = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                processed = np.uint8(np.absolute(processed))
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif edge_type == "Prewitt":
                kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                img_prewittx = cv2.filter2D(gray, -1, kernelx)
                img_prewitty = cv2.filter2D(gray, -1, kernely)
                processed = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:  # Roberts
                kernelx = np.array([[1, 0], [0, -1]])
                kernely = np.array([[0, 1], [-1, 0]])
                img_robertsx = cv2.filter2D(gray, -1, kernelx)
                img_robertsy = cv2.filter2D(gray, -1, kernely)
                processed = cv2.addWeighted(img_robertsx, 0.5, img_robertsy, 0.5, 0)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        elif processor == "Color Adjustments":
            adjust_type = st.radio("Adjustment Type:", 
                                 ["Brightness/Contrast", "HSV Adjustment", "Color Balance", "Color Grading", "Auto Adjust"])
            
            if adjust_type == "Brightness/Contrast":
                alpha = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)  # Contrast control
                beta = st.slider("Brightness", -100, 100, 0)  # Brightness control
                processed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            elif adjust_type == "HSV Adjustment":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                h_shift = st.slider("Hue Shift", -180, 180, 0)
                s_scale = st.slider("Saturation Scale", 0.0, 2.0, 1.0, 0.1)
                v_scale = st.slider("Value (Brightness) Scale", 0.0, 2.0, 1.0, 0.1)
                
                # Apply adjustments
                hsv[:,:,0] = (hsv[:,:,0] + h_shift) % 180
                hsv[:,:,1] = np.clip(hsv[:,:,1] * s_scale, 0, 255).astype(np.uint8)
                hsv[:,:,2] = np.clip(hsv[:,:,2] * v_scale, 0, 255).astype(np.uint8)
                processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            elif adjust_type == "Color Balance":
                r_scale = st.slider("Red Channel", 0.0, 2.0, 1.0, 0.1)
                g_scale = st.slider("Green Channel", 0.0, 2.0, 1.0, 0.1)
                b_scale = st.slider("Blue Channel", 0.0, 2.0, 1.0, 0.1)
                
                # Split channels and scale them
                b, g, r = cv2.split(image)
                b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
                g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
                r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
                processed = cv2.merge([b, g, r])
                
            elif adjust_type == "Color Grading":
                preset = st.selectbox("Preset", ["Warm", "Cool", "Vintage", "Cinematic", "High Contrast"])
                
                if preset == "Warm":
                    processed = cv2.convertScaleAbs(image, alpha=1.05, beta=10)
                    b, g, r = cv2.split(processed)
                    r = np.clip(r * 1.2, 0, 255).astype(np.uint8)
                    processed = cv2.merge([b, g, r])
                elif preset == "Cool":
                    processed = cv2.convertScaleAbs(image, alpha=1.05, beta=0)
                    b, g, r = cv2.split(processed)
                    b = np.clip(b * 1.2, 0, 255).astype(np.uint8)
                    processed = cv2.merge([b, g, r])
                elif preset == "Vintage":
                    processed = cv2.convertScaleAbs(image, alpha=0.9, beta=10)
                    b, g, r = cv2.split(processed)
                    r = np.clip(r * 1.1, 0, 255).astype(np.uint8)
                    g = np.clip(g * 0.9, 0, 255).astype(np.uint8)
                    processed = cv2.merge([b, g, r])
                elif preset == "Cinematic":
                    processed = cv2.convertScaleAbs(image, alpha=1.2, beta=-10)
                    hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
                    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
                    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:  # High Contrast
                    processed = cv2.convertScaleAbs(image, alpha=1.5, beta=-20)
                
            else:  # Auto Adjust
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                min_val, max_val = np.percentile(gray, [2, 98])
                alpha = 255 / (max_val - min_val)
                beta = -min_val * alpha
                processed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                
                # Optional auto white balance
                if st.checkbox("Auto White Balance", value=True):
                    b, g, r = cv2.split(processed)
                    r_avg = np.mean(r)
                    g_avg = np.mean(g)
                    b_avg = np.mean(b)
                    avg = (r_avg + g_avg + b_avg) / 3
                    r = np.clip(r * (avg / r_avg), 0, 255).astype(np.uint8)
                    g = np.clip(g * (avg / g_avg), 0, 255).astype(np.uint8)
                    b = np.clip(b * (avg / b_avg), 0, 255).astype(np.uint8)
                    processed = cv2.merge([b, g, r])
        
        elif processor == "Transformations":
            transform_type = st.radio("Transformation Type:", 
                                    ["Resize", "Rotate", "Flip", "Crop", "Perspective"])
            
            if transform_type == "Resize":
                scale = st.slider("Scale Factor", 0.1, 2.0, 1.0, 0.1)
                interpolation = st.selectbox("Interpolation", 
                                          ["Nearest", "Bilinear", "Bicubic", "Lanczos"])
                
                interp_methods = {
                    "Nearest": cv2.INTER_NEAREST,
                    "Bilinear": cv2.INTER_LINEAR,
                    "Bicubic": cv2.INTER_CUBIC,
                    "Lanczos": cv2.INTER_LANCZOS4
                }
                
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                processed = cv2.resize(image, (new_width, new_height), 
                                     interpolation=interp_methods[interpolation])
                
            elif transform_type == "Rotate":
                angle = st.slider("Angle (degrees)", -180, 180, 0)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
                
            elif transform_type == "Flip":
                flip_direction = st.radio("Direction", ["Horizontal", "Vertical", "Both"])
                
                if flip_direction == "Horizontal":
                    processed = cv2.flip(image, 1)
                elif flip_direction == "Vertical":
                    processed = cv2.flip(image, 0)
                else:  # Both
                    processed = cv2.flip(image, -1)
                    
            elif transform_type == "Crop":
                col1, col2 = st.columns(2)
                with col1:
                    x1 = st.slider("X Start", 0, image.shape[1]-10, int(image.shape[1]*0.25))
                    y1 = st.slider("Y Start", 0, image.shape[0]-10, int(image.shape[0]*0.25))
                with col2:
                    x2 = st.slider("X End", x1+10, image.shape[1], int(image.shape[1]*0.75))
                    y2 = st.slider("Y End", y1+10, image.shape[0], int(image.shape[0]*0.75))
                
                processed = image[y1:y2, x1:x2].copy()
                
            else:  # Perspective
                # Default points (corners of the image)
                h, w = image.shape[:2]
                st.write("Adjust perspective points by moving sliders")
                
                col1, col2 = st.columns(2)
                with col1:
                    x1 = st.slider("Top Left X", 0, w//2, 0)
                    y1 = st.slider("Top Left Y", 0, h//2, 0)
                    x2 = st.slider("Bottom Left X", 0, w//2, 0)
                    y2 = st.slider("Bottom Left Y", h//2, h, h)
                
                with col2:
                    x3 = st.slider("Top Right X", w//2, w, w)
                    y3 = st.slider("Top Right Y", 0, h//2, 0)
                    x4 = st.slider("Bottom Right X", w//2, w, w)
                    y4 = st.slider("Bottom Right Y", h//2, h, h)
                
                # Source points
                pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
                # Destination points
                pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                
                # Get perspective transform matrix
                M = cv2.getPerspectiveTransform(pts1, pts2)
                # Apply perspective transformation
                processed = cv2.warpPerspective(image, M, (w, h))
        
        elif processor == "Special Effects":
            effect_type = st.selectbox("Effect Type:", 
                                     ["Black & White", "Sepia", "Negative", "Stylization", 
                                      "Pencil Sketch", "Cartoon", "Emboss", "Vignette"])
            
            if effect_type == "Black & White":
                method = st.radio("Method", ["Simple", "Weighted", "Adaptive"])
                
                if method == "Simple":
                    processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                elif method == "Weighted":
                    weights = np.array([0.299, 0.587, 0.114])
                    grayscale = np.dot(image[:, :, :3], weights)
                    processed = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)
                else:  # Adaptive
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    block_size = st.slider("Block Size", 3, 99, 11, 2)
                    C = st.slider("C Value", -30, 30, 2)
                    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, block_size, C)
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                
            elif effect_type == "Sepia":
                strength = st.slider("Strength", 0.0, 1.0, 0.5, 0.1)
                
                # Convert to float32
                img_float = np.array(image, dtype=np.float32) / 255.0
                
                # Sepia matrix
                sepia_matrix = np.array([
                    [0.393 + 0.607 * (1 - strength), 0.769 - 0.769 * (1 - strength), 0.189 - 0.189 * (1 - strength)],
                    [0.349 - 0.349 * (1 - strength), 0.686 + 0.314 * (1 - strength), 0.168 - 0.168 * (1 - strength)],
                    [0.272 - 0.272 * (1 - strength), 0.534 - 0.534 * (1 - strength), 0.131 + 0.869 * (1 - strength)]
                ])
                
                # Apply sepia effect
                sepia_img = np.zeros_like(img_float)
                for i in range(3):
                    sepia_img[:, :, i] = np.sum(img_float * sepia_matrix[i, :].reshape(1, 1, 3), axis=2)
                
                # Convert back to uint8
                processed = np.clip(sepia_img * 255, 0, 255).astype(np.uint8)
                
            elif effect_type == "Negative":
                processed = 255 - image
                
            elif effect_type == "Stylization":
                sigma_s = st.slider("Spatial Sigma", 10, 200, 60)
                sigma_r = st.slider("Range Sigma", 1, 100, 40) / 100.0
                processed = cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
                
            elif effect_type == "Pencil Sketch":
                sigma_s = st.slider("Smoothness", 1, 200, 60)
                sigma_r = st.slider("Gradient Preservation", 1, 100, 40) / 100.0
                shade_factor = st.slider("Shading Factor", 0.0, 0.5, 0.1, 0.05)
                
                gray, color = cv2.pencilSketch(image, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
                
                sketch_type = st.radio("Type", ["Grayscale", "Color"])
                if sketch_type == "Grayscale":
                    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    processed = color
                
            elif effect_type == "Cartoon":
                # Adapted from: https://www.geeksforgeeks.org/cartooning-an-image-using-opencv-python/
                # Edge Detection
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 9, 9)
                
                # Color Quantization
                k = st.slider("Color Levels", 2, 16, 8)
                data = np.float32(image).reshape((-1, 3))
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
                _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                result = center[label.flatten()]
                result = result.reshape(image.shape)
                
                # Combine with edges
                if st.checkbox("Show Edges", value=True):
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    processed = cv2.bitwise_and(result, edges_rgb)
                else:
                    processed = result
                
            elif effect_type == "Emboss":
                kernel = np.array([[0, -1, -1],
                                 [1, 0, -1],
                                 [1, 1, 0]])
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                emboss = cv2.filter2D(gray, -1, kernel) + 128
                processed = cv2.cvtColor(emboss.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                
            else:  # Vignette
                rows, cols = image.shape[:2]
                
                # Vignette parameters
                sigma = st.slider("Size", 0.1, 2.0, 0.5, 0.1)
                strength = st.slider("Darkness", 0.0, 1.0, 0.5, 0.05)
                
                # Generate vignette mask
                kernel_x = cv2.getGaussianKernel(cols, sigma * cols)
                kernel_y = cv2.getGaussianKernel(rows, sigma * rows)
                kernel = kernel_y * kernel_x.T
                mask = 255 * (kernel / np.max(kernel))
                mask = np.power(mask, strength)
                
                # Apply vignette
                processed = image.copy()
                for i in range(3):
                    processed[:, :, i] = processed[:, :, i] * mask / 255.0
                
                processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # Display processed image
        st.image(processed, use_column_width=True)
        
        # Statistics for processed image
        with st.expander("Processed Image Statistics", expanded=False):
            stats = display_stats(processed)
            stat_cols = st.columns(4)
            for i, (label, value) in enumerate(stats.items()):
                with stat_cols[i]:
                    st.metric(label, value)
    
    # Side by side comparison if enabled
    if show_comparison:
        st.markdown("### Before & After Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.image(image, caption="Original", use_column_width=True)
        
        with comp_col2:
            st.image(processed, caption="Processed", use_column_width=True)
    
    # Download button for processed image
    buf = cv2.imencode('.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.sidebar.download_button(
        label="Download Processed Image",
        data=buf,
        file_name="processed_image.png",
        mime="image/png"
    )

else:
    # Show placeholder when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px">
        <h2 style="color: #666">📁 Drag & Drop Image to Begin</h2>
        <p style="color: #444">Supports JPG, PNG, JPEG formats</p>
    </div>
    """, unsafe_allow_html=True)