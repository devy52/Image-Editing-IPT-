import streamlit as st
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
import numpy as np
from skimage import img_as_float
import io
import base64

def invert_img(img):
    inverted_image = img.point(lambda x: 255 - x)
    return inverted_image

def adjust_contrast(img, factor=2.0):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndi.convolve(img, Kx)
    Iy = ndi.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return G, theta

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    
    return Z

def rotate_image(img, degrees):
    rotated_img = ndi.rotate(img, degrees, reshape=False)
    return rotated_img

def get_image_download_link(processed_img, filename, text):
    if isinstance(processed_img, np.ndarray):
        # Convert NumPy array to PIL Image
        processed_img = Image.fromarray(processed_img.astype(np.uint8))
    
    buffered = io.BytesIO()
    processed_img.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    b64 = base64.b64encode(img_str).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Streamlit UI
st.title("Image Editing App")

# Sidebar for user interactions
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Brightness adjustment
brightness_factor = st.sidebar.slider("Adjust brightness", 0.1, 3.0, 1.0)

# Calibration brightness adjustment (for original image)
calibration_brightness_factor = st.sidebar.slider("Calibration brightness", 0.1, 3.0, 1.0)

# Rotation control with buttons (0°, 90°, 180°, 270°)
rotation_angle = st.sidebar.radio("Rotate image", [0, 90, 180, 270])

# Processing options
options = ["Original", "Grayscale", "Sketch", "Rough Drawing"]
option = st.sidebar.selectbox("Choose processing step", options)

if uploaded_file is not None:
    # Load image
    original_image = Image.open(uploaded_file)
    st.sidebar.image(original_image, caption="Uploaded Image", use_column_width=True)
    
    # Apply brightness adjustment to a copy of the original image
    calibrated_image = ImageEnhance.Brightness(original_image).enhance(calibration_brightness_factor)
    
    # Apply brightness adjustment
    image = ImageEnhance.Brightness(calibrated_image).enhance(brightness_factor)
    
    # Convert to float
    img = img_as_float(image)
    
    # Process based on selected option
    if option == "Original":
        processed_img = original_image
        
    elif option == "Grayscale":
        gray_img = rgb2gray(img)
        grayscale_image_display = (gray_img * 255).astype(np.uint8)
        processed_img = Image.fromarray(grayscale_image_display)
        
    elif option == "Sketch":
        gray_img = rgb2gray(img)
        b, c = sobel_filters(gray_img)
        sobel_image_display = b.astype(np.uint8)
        inverted_sobel_image = invert_img(Image.fromarray(sobel_image_display))
        contrast_inverted_sobel_image = adjust_contrast(inverted_sobel_image)
        processed_img = contrast_inverted_sobel_image
        
    elif option == "Rough Drawing":
        gray_img = rgb2gray(img)
        b, c = sobel_filters(gray_img)
        d = non_max_suppression(b, c)
        nms_range = np.ptp(d)
        if nms_range != 0:
            nms_image_display = (d - np.min(d)) / nms_range
        else:
            nms_image_display = d  # No normalization if range is zero
        nms_image_display = (nms_image_display * 255).astype(np.uint8)
        inverted_nms_image = invert_img(Image.fromarray(nms_image_display))
        contrast_inverted_nms_image = adjust_contrast(inverted_nms_image)
        processed_img = contrast_inverted_nms_image
    
    # Rotate the processed image if rotation angle is not zero
    if rotation_angle != 0:
        processed_img = rotate_image(processed_img, rotation_angle)
    
    # Display the processed and rotated image
    st.image(processed_img, caption=f"{option} Result (Rotated {rotation_angle}°)", use_column_width=True)
    
    # Download button
    if st.button('Download Result Image'):
        download_link = get_image_download_link(processed_img, f'{option}_result.png', 'Click here to download')
        st.markdown(download_link, unsafe_allow_html=True)
