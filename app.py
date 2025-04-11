import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure page
st.set_page_config(page_title="Image Processing", layout="wide", page_icon="🔬")
st.title("Image Processing")

# Custom CSS for blue and yellow UI
st.markdown("""
<style>
    /* Blue background for main area */
    .stApp {
        background-color: #1e3d59;
        color: #f5f0e1;
    }
    
    /* Yellow accents */
    .stButton button, .stDownloadButton button {
        background-color: #ffc13b !important;
        color: #1e3d59 !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    /* Styling for containers */
    .metric-card {
        background-color: #2b5d8a;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        margin: 10px 0;
        color: #f5f0e1;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffc13b !important;
    }
    
    /* Control elements styling */
    .stRadio > div {
        background-color: #2b5d8a;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        color: #ffc13b !important;
    }
    
    /* Bottom feature bar */
    .feature-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0f2942;
        padding: 15px;
        z-index: 999;
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Main content area
main_container = st.container()

with main_container:
    # Image upload with drag & drop zone
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 📤 Upload Image")
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
        display_col1, display_col2 = st.columns(2, gap="large")
        
        with display_col1:
            st.markdown("### Original Image")
            st.image(image, use_column_width=True)
            
            with st.expander("Original Image Statistics", expanded=False):
                stats = display_stats(image)
                stat_cols = st.columns(4)
                for i, (label, value) in enumerate(stats.items()):
                    with stat_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{label}</h4>
                            <p>{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Processing pipeline
        processed = image.copy()
        
        # Create the bottom feature bar (will be placed first in DOM but styled to appear at bottom)
        st.markdown('<div class="feature-bar">', unsafe_allow_html=True)
        feature_columns = st.columns([1, 1, 1, 1, 1])
        
        with feature_columns[0]:
            processor = st.radio("Operation", [
                "Smoothing",
                "Sharpening",
                "Edge Detection",
                "Color Adjustments"
            ], horizontal=True)
        
        # Based on selection, show relevant controls
        with feature_columns[1:]:
            if processor == "Smoothing":
                with feature_columns[1]:
                    smooth_type = st.radio("Filter Type", ["Gaussian", "Median", "Bilateral"])
                with feature_columns[2]:
                    kernel_size = st.slider("Kernel Size", 3, 25, 9, 2)
                
                if smooth_type == "Gaussian":
                    with feature_columns[3]:
                        sigma = st.slider("Sigma", 0.1, 5.0, 1.5)
                    processed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
                elif smooth_type == "Median":
                    processed = cv2.medianBlur(image, kernel_size)
                else:
                    with feature_columns[3]:
                        d = st.slider("Diameter", 1, 15, 9)
                    with feature_columns[4]:
                        sigma_color = st.slider("Color Sigma", 1, 200, 75)
                        sigma_space = st.slider("Spatial Sigma", 1, 200, 75)
                    processed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            elif processor == "Sharpening":
                with feature_columns[1]:
                    sharp_type = st.radio("Technique", ["Laplacian", "Unsharp Mask"])
                
                if sharp_type == "Laplacian":
                    with feature_columns[2]:
                        strength = st.slider("Strength", 1, 10, 5) 
                    kernel = np.array([[0, -1, 0], [-1, strength, -1], [0, -1, 0]])
                    processed = cv2.filter2D(image, -1, kernel)
                else:
                    with feature_columns[2]:
                        strength = st.slider("Strength", 0.5, 3.0, 1.5)
                    blur = cv2.GaussianBlur(image, (0,0), 3)
                    processed = cv2.addWeighted(image, strength, blur, -0.5, 0)
            
            elif processor == "Edge Detection":
                with feature_columns[1]:
                    edge_type = st.selectbox("Method", ["Canny", "Sobel", "Laplacian"])
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                if edge_type == "Canny":
                    with feature_columns[2]:
                        threshold1 = st.slider("Low Threshold", 0, 255, 50)
                    with feature_columns[3]:
                        threshold2 = st.slider("High Threshold", 0, 255, 150)
                    processed = cv2.Canny(gray, threshold1, threshold2)
                    # Convert back to 3 channels for consistent display
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                elif edge_type == "Sobel":
                    with feature_columns[2]:
                        dx = st.slider("X Derivative", 0, 2, 1)
                    with feature_columns[3]:
                        dy = st.slider("Y Derivative", 0, 2, 1)
                    with feature_columns[4]:
                        ksize = st.slider("Kernel Size", 1, 7, 5, 2)
                    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
                    processed = np.uint8(np.absolute(sobel))
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                else:
                    with feature_columns[2]:
                        ksize = st.slider("Kernel Size", 1, 7, 3, 2)
                    processed = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                    processed = np.uint8(np.absolute(processed))
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            
            elif processor == "Color Adjustments":
                with feature_columns[1]:
                    adjust_type = st.radio("Adjustment", ["Brightness/Contrast", "HSV", "Color Balance"])
                
                if adjust_type == "Brightness/Contrast":
                    with feature_columns[2]:
                        alpha = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)  # Contrast control
                    with feature_columns[3]:
                        beta = st.slider("Brightness", -100, 100, 0)  # Brightness control
                    processed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                
                elif adjust_type == "HSV":
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    with feature_columns[2]:
                        h_shift = st.slider("Hue", -180, 180, 0)
                    with feature_columns[3]:
                        s_scale = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
                    with feature_columns[4]:
                        v_scale = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
                    
                    # Apply adjustments
                    hsv[:,:,0] = (hsv[:,:,0] + h_shift) % 180
                    hsv[:,:,1] = np.clip(hsv[:,:,1] * s_scale, 0, 255).astype(np.uint8)
                    hsv[:,:,2] = np.clip(hsv[:,:,2] * v_scale, 0, 255).astype(np.uint8)
                    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                elif adjust_type == "Color Balance":
                    with feature_columns[2]:
                        r_scale = st.slider("Red", 0.0, 2.0, 1.0, 0.1)
                    with feature_columns[3]:
                        g_scale = st.slider("Green", 0.0, 2.0, 1.0, 0.1)
                    with feature_columns[4]:
                        b_scale = st.slider("Blue", 0.0, 2.0, 1.0, 0.1)
                    
                    # Split channels and scale them
                    b, g, r = cv2.split(image)
                    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
                    g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
                    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
                    processed = cv2.merge([b, g, r])
        
        # Close the feature bar div
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display processed image
        with display_col2:
            st.markdown("### Processed Image")
            st.image(processed, use_column_width=True)
            
            with st.expander("Processed Image Statistics", expanded=False):
                stats = display_stats(processed)
                stat_cols = st.columns(4)
                for i, (label, value) in enumerate(stats.items()):
                    with stat_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{label}</h4>
                            <p>{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Side by side comparison
        st.markdown("### Before & After Comparison")
        show_comparison = st.checkbox("Show Side-by-Side Comparison", value=True)
        
        if show_comparison:
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.image(image, caption="Original", use_column_width=True)
            
            with comp_col2:
                st.image(processed, caption="Processed", use_column_width=True)
        
        # Download button for processed image
        buf = cv2.imencode('.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))[1].tobytes()
        with col2:
            st.download_button(
                label="⬇️ Download Processed Image",
                data=buf,
                file_name="processed_image.png",
                mime="image/png"
            )

    else:
        # Show placeholder when no image is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 100px 20px; background-color: #2b5d8a; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: #ffc13b">📁 Drag & Drop Image to Begin</h2>
            <p style="color: #f5f0e1">Supports JPG, PNG, JPEG formats</p>
        </div>
        """, unsafe_allow_html=True)