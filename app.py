import streamlit as st
import processor
import numpy as np

st.set_page_config(page_title="Background Remover", layout="centered")

st.title("去图片背景小程序")
st.write("上传一张带纯色背景的图片，调整阈值以去除背景。")

uploaded_file = st.file_uploader("上传图片 (PNG, JPG)", type=["png", "jpg", "jpeg"])

threshold = st.slider("去背阈值 (Threshold)", min_value=0, max_value=255, value=20, help="值越大，容忍的颜色差异越大。")

if uploaded_file is not None:
    # Read bytes
    image_bytes = uploaded_file.getvalue()
    
    try:
        # Process
        result_bytes, original_img, processed_img = processor.process_image(image_bytes, threshold)
        
        # Display Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("原始图片")
            # Convert BGR to RGB for display
            st.image(original_img[:, :, ::-1], use_container_width=True)
            
        with col2:
            st.header("处理结果")
            # Processed is RGBA. Streamlit handles it fine.
            # But wait, original was cv2 image (BGR). processed is RGBA?
            # Let's check processor.py.
            # Yes, rgba = cv2.merge([b, g, r, alpha_channel]).
            # Streamlit expects RGB or RGBA.
            # If we pass numpy array to st.image, it assumes RGB(A).
            # But cv2 uses BGR. 
            # So for `rgba`, the channels are B, G, R, A.
            # We need to convert B,G,R,A -> R,G,B,A for st.image
            
            # Convert BGRA to RGBA
            processed_display = processed_img.copy()
            processed_display[:, :, 0] = processed_img[:, :, 2] # R = R
            processed_display[:, :, 2] = processed_img[:, :, 0] # B = B
            
            st.image(processed_display, use_container_width=True)
            
        st.success("处理完成！")
        
        # Download Button
        st.download_button(
            label="下载去背图片 (PNG)",
            data=result_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )
            
    except Exception as e:
        st.error(f"发生错误: {e}")
