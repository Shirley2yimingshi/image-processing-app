import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from filters import apply_filters, fft_spectrum, draw_gradient_arrows

st.title("Image Processing")

uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("图片读取失败")
    else:
        st.image(img, caption="原图", channels="BGR")

        mode = st.selectbox(
            "选择功能",
            ["Filter", "Gradient", "FFT"]
        )


        # Filter
        if mode == "Filter":
            st.subheader("滤波对比")

            k = st.slider("Kernel Size", 3, 15, 5, step=2)

            filters = {
                "box": cv2.blur(img, (k, k)),
                "gaussian": cv2.GaussianBlur(img, (k, k), 0),
                "median": cv2.medianBlur(img, k)
            }

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            col1.image(filters["box"], caption="Box", channels="BGR")
            col2.image(filters["gaussian"], caption="Gaussian", channels="BGR")
            col3.image(filters["median"], caption="Median", channels="BGR")
            col4.image(sobel, caption="Sobel", clamp=True)


        # Gradient
        elif mode == "Gradient":
            st.subheader("框选区域计算梯度")
        
            # ⭐ 关键：先缩放图像（解决canvas错位）
            max_size = 600
            h, w = img.shape[:2]
            scale = max_size / max(h, w)
        
            new_w = int(w * scale)
            new_h = int(h * scale)
        
            img_resized = cv2.resize(img, (new_w, new_h))
        
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                background_image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)),
                update_streamlit=True,
                height=new_h,
                width=new_w,
                drawing_mode="rect"
            )
        
            if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                rect = canvas_result.json_data["objects"][-1]
        
                # canvas坐标
                x = int(rect["left"])
                y = int(rect["top"])
                w_box = int(rect["width"])
                h_box = int(rect["height"])
        
                # 转回原图坐标
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w_box / scale)
                h_orig = int(h_box / scale)
        
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                patch = gray[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
        
                if patch.size == 0:
                    st.warning("选区无效")
                    st.stop()
        
                # 计算梯度
                gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0)
                gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1)
        
                magnitude = np.sqrt(gx**2 + gy**2)
                direction = np.arctan2(gy, gx)
        
                # 归一化为 uint8
                magnitude_disp = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                direction_disp = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
                # 绘制梯度箭头
                arrows = draw_gradient_arrows(magnitude_disp, gx, gy, step=10, scale=0.3)
        
                # 显示结果
                st.image(patch, caption="选中区域", clamp=True)
                st.image(magnitude_disp, caption="梯度幅值", clamp=True)
                st.image(direction_disp, caption="梯度方向", clamp=True)
                st.image(arrows, caption="箭头方向图", channels="BGR")
               
        # FFT
        else:
            st.subheader("频域分析")

            def norm(x):
                return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            _, spec_orig = fft_spectrum(img)

            M = np.float32([[1, 0, 50], [0, 1, 50]])
            translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            _, spec_trans = fft_spectrum(translated)

            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            _, spec_rot = fft_spectrum(rotated)

            scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
            _, spec_scale = fft_spectrum(scaled)

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            col1.image(norm(spec_orig), caption="原图频谱", clamp=True)
            col2.image(norm(spec_trans), caption="平移频谱", clamp=True)
            col3.image(norm(spec_rot), caption="旋转频谱", clamp=True)
            col4.image(norm(spec_scale), caption="缩放频谱", clamp=True)
