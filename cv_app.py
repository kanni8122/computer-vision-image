import cv2
import PIL
from PIL import Image
import numpy as np
import streamlit as st

print("OpenCV version:", cv2.__version__)

try:
    haarcascades_path = cv2.data.haarcascades
    print("Haarcascades path:", haarcascades_path)
except AttributeError as e:
    print("AttributeError:", e)


st.title("Advanced Computer Vision Project")
st.write("Upload an image and apply various transformations.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with st.expander("Original Image"):
        st.image(img_array, caption='Original Image', use_column_width=True)

    with st.expander("GrayScale Image"):
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        st.image(gray_image, caption='Grayscale Image', use_column_width=True, channels='GRAY')

    with st.expander("Resize Image"):
        st.write("### Resize Image")
        width = st.slider("Width", min_value=10, max_value=img_array.shape[1], value=img_array.shape[1])
        height = st.slider("Height", min_value=10, max_value=img_array.shape[0], value=img_array.shape[0])
        resized_image = cv2.resize(img_array, (width, height))
        st.image(resized_image, caption='Resized Image', use_column_width=True)

    with st.expander("Edge Detection"):
        st.write("### Edge Detection")
        low_threshold = st.slider("Low Threshold", 0, 255, 100)
        high_threshold = st.slider("High Threshold", 0, 255, 200)
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        st.image(edges, caption='Edge Detection', use_column_width=True, channels='GRAY')

    with st.expander("Gaussian Blurring"):
        st.write("### Gaussian Blur")
        ksize = st.slider("Kernel Size", 1, 15, 3, step=2)
        if ksize % 2 == 0:
            ksize += 1
        blurred_image = cv2.GaussianBlur(img_array, (ksize, ksize), 0)
        st.image(blurred_image, caption='Blurred Image', use_column_width=True)

    with st.expander("Brightness and Contrast Adjuster"):
        st.write("### Brightness and Contrast Adjustment")
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)
        adjusted_image = cv2.convertScaleAbs(img_array, alpha=1 + contrast / 100.0, beta=brightness)
        st.image(adjusted_image, caption='Brightness and Contrast Adjusted Image', use_column_width=True)

    with st.expander("Rotation Of Image"):
        st.write("### Rotate Image")
        angle = st.slider("Rotation Angle", -180, 180, 0)
        (h, w) = img_array.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img_array, M, (w, h))
        st.image(rotated_image, caption='Rotated Image', use_column_width=True)

    with st.expander("Histogram Equalization"):
        st.write("### Histogram Equalization")
        equalized_image = cv2.equalizeHist(gray_image)
        st.image(equalized_image, caption='Histogram Equalized Image', use_column_width=True, channels='GRAY')

    with st.expander("Image Filtering"):
        st.write("### Image Filtering")
        filter_type = st.selectbox("Choose a filter", ["None", "Median", "Bilateral"])
        if filter_type == "Median":
            ksize = st.slider("Kernel Size for Median Filter", 1, 15, 3, step=2)
            if ksize % 2 == 0:
                ksize += 1
            filtered_image = cv2.medianBlur(img_array, ksize)
            st.image(filtered_image, caption='Median Filtered Image', use_column_width=True)
        elif filter_type == "Bilateral":
            d = st.slider("Diameter", 1, 15, 9)
            sigma_color = st.slider("Sigma Color", 1, 100, 75)
            sigma_space = st.slider("Sigma Space", 1, 100, 75)
            filtered_image = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
            st.image(filtered_image, caption='Bilateral Filtered Image', use_column_width=True)

    with st.expander("Face Detection"):
        st.write("### Face Detection")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        face_detected_image = img_array.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(face_detected_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.image(face_detected_image, caption='Face Detection', use_column_width=True)

    with st.expander("Color Space Conversion"):
        st.write("### Color Space Conversion")
        color_space = st.selectbox("Choose a color space", ["None", "HSV", "LAB"])
        if color_space == "HSV":
            hsv_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
            st.image(hsv_image, caption='HSV Image', use_column_width=True)
        elif color_space == "LAB":
            lab_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
            st.image(lab_image, caption='LAB Image', use_column_width=True)

    with st.expander("Contour Detection"):
        st.write("### Contour Detection")
        ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.drawContours(img_array.copy(), contours, -1, (0, 255, 0), 3)
        st.image(contour_image, caption='Contour Detection', use_column_width=True)

    with st.expander("Image Sharpening"):
        st.write("### Image Sharpening")
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(img_array, -1, kernel)
        st.image(sharpened_image, caption='Sharpened Image', use_column_width=True)






