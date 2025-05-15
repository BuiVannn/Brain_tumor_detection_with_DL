import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import io
import os

# Cấu hình trang
st.set_page_config(
    page_title="Hệ Thống Phân Loại Khối U Não MRI",
    page_icon="🧠",
    layout="wide"
)

# Tiêu đề và thông tin
st.title("🧠 Hệ Thống Phân Loại Khối U Não MRI")
st.markdown("""
    Hệ thống này phân loại ảnh MRI não thành 4 nhóm: **Glioma**, **Meningioma**, **No_Tumor** (Không có u) và **Pituitary** (U tuyến yên).
    
    Tải lên ảnh MRI để nhận kết quả phân loại.
""")

# Hàm tải mô hình
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_resnet_attention_model.h5')
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return None

# Hàm tiền xử lý ảnh
def preprocess_image(image, target_size=(224, 224)):
    # Chuyển đổi sang mảng numpy nếu là PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Đảm bảo hình ảnh là RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize về kích thước 224x224 
    image = cv2.resize(image, target_size)
    
    # Chuẩn hóa về giá trị [0,1]
    image = image.astype('float32') / 255.0
    
    return image

# Hàm dự đoán
def predict(model, image):
    # Tiền xử lý ảnh
    processed_img = preprocess_image(image)
    
    # Mở rộng kích thước batch
    img_batch = np.expand_dims(processed_img, axis=0)
    
    # Dự đoán
    start_time = time.time()
    prediction = model.predict(img_batch)
    end_time = time.time()
    
    # Lấy class_id có xác suất cao nhất
    predicted_class_id = np.argmax(prediction, axis=1)[0]
    
    # Ánh xạ các lớp
    class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'No_Tumor', 3: 'Pituitary'}
    predicted_class = class_names[predicted_class_id]
    
    # Tính xác suất cho mỗi lớp
    probabilities = prediction[0]
    
    # Thời gian xử lý
    processing_time = end_time - start_time
    
    return predicted_class, probabilities, processing_time, processed_img

# Tạo visualize CAM (Class Activation Map) để xem model chú ý vào vùng nào
def get_grad_cam(model, image, layer_name='attention_block'):
    # Tiền xử lý ảnh
    img_array = preprocess_image(image)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    try:
        # Tìm lớp attention hoặc lớp cuối của ConvNet
        last_conv_layer = None
        for layer in model.layers:
            if layer_name in layer.name:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Nếu không tìm thấy attention block, dùng lớp conv cuối cùng
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            return None
        
        # Tạo model để lấy feature maps từ lớp cuối
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Gradient tape để tính gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]
            
        # Gradient của output đối với feature maps
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Nhân feature maps với gradient weights
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Chuẩn hóa heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap về kích thước ảnh gốc
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Chuyển heatmap thành màu
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Chuyển ảnh gốc về đúng định dạng
        superimposed_img = np.uint8(img_array * 255)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        
        # Kết hợp ảnh gốc và heatmap
        superimposed_img = heatmap * 0.4 + superimposed_img
        
        # Chuyển về RGB để hiển thị
        superimposed_img = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)
        
        return superimposed_img
        
    except Exception as e:
        st.error(f"Lỗi khi tạo Grad-CAM: {e}")
        return None

# Tải mô hình
model = load_model()

# Tạo layout 2 cột
col1, col2 = st.columns([1, 1])

# Cột bên trái để upload ảnh
with col1:
    st.subheader("Tải lên ảnh MRI não")
    uploaded_file = st.file_uploader("Chọn ảnh MRI", type=["jpg", "jpeg", "png"])
    
    # Tùy chọn nâng cao (có thể ẩn đi mặc định)
    with st.expander("Tùy chọn nâng cao", expanded=False):
        show_cam = st.checkbox("Hiển thị bản đồ nhiệt (CAM)", value=True)
    
    # Nút dự đoán
    predict_button = st.button("Phân loại ảnh")

# Cột bên phải để hiển thị kết quả
with col2:
    st.subheader("Kết quả phân loại")
    
    if uploaded_file is not None:
        # Hiển thị ảnh đã tải lên
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
        
        if predict_button:
            if model is not None:
                # Dự đoán
                with st.spinner('Đang phân loại ảnh...'):
                    try:
                        predicted_class, probabilities, processing_time, processed_img = predict(model, image)
                        
                        # Hiển thị kết quả
                        st.success(f"**Kết quả phân loại:** {predicted_class}")
                        st.info(f"Thời gian xử lý: {processing_time:.4f} giây")
                        
                        # Hiển thị các xác suất dưới dạng thanh tiến trình
                        st.subheader("Xác suất của các lớp:")
                        class_names = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']
                        for i, cls in enumerate(class_names):
                            prob = probabilities[i] * 100
                            st.metric(
                                label=cls,
                                value=f"{prob:.2f}%",
                                delta=f"{prob - 25:.2f}%" if prob > 25 else f"{prob - 25:.2f}%"
                            )
                            st.progress(float(prob) / 100)
                            
                        # Hiển thị bản đồ nhiệt (Grad-CAM)
                        if show_cam:
                            st.subheader("Vùng chú ý của mô hình (Grad-CAM):")
                            with st.spinner('Đang tạo bản đồ nhiệt...'):
                                cam_image = get_grad_cam(model, image)
                                if cam_image is not None:
                                    st.image(cam_image, caption="Bản đồ nhiệt cho vùng quan trọng", use_column_width=True)
                                else:
                                    st.warning("Không thể tạo bản đồ nhiệt cho ảnh này.")
                                    
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                        st.exception(e)
            else:
                st.error("Không thể tải mô hình. Vui lòng kiểm tra lại đường dẫn đến file mô hình.")

# Hiển thị thông tin về mô hình và lớp
with st.expander("Thông tin về các loại khối u", expanded=False):
    st.markdown("""
    ### Các loại khối u não:
    
    1. **Glioma**: 
       - Là khối u phát sinh từ tế bào hỗ trợ (tế bào đệm) trong não
       - Có thể là lành tính hoặc ác tính
       - Bao gồm các loại như u sao bào, u thần kinh đệm và u nguyên bào đệm
    
    2. **Meningioma**:
       - Phát triển từ màng não (màng bao quanh não và tủy sống)
       - Thường là lành tính và phát triển chậm
       - Có thể gây áp lực lên não và các dây thần kinh
    
    3. **Pituitary** (U tuyến yên):
       - Phát triển ở tuyến yên, một tuyến nhỏ nằm ở đáy não
       - Thường là lành tính
       - Có thể ảnh hưởng đến sự sản xuất hormone và gây ra các vấn đề về nội tiết
    
    ### Phương pháp phân loại:
    
    Mô hình này sử dụng mạng nơ-ron tích chập (CNN) với kiến trúc ResNet50V2 kết hợp cơ chế Attention để phân loại chính xác các loại khối u não từ ảnh MRI.
    """)

# Footer
st.markdown("""
---
**Lưu ý:** Kết quả phân loại này chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán y tế chuyên nghiệp.
""")