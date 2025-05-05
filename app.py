import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2

st.set_page_config(page_title="Flower Classifier", layout="wide")
st.title("Классификатор цветов")

API_URL = "https://db32-34-87-103-23.ngrok-free.app/predict"

# Список классов
CLASSES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

def preprocess_image(image):
    """Предобработка изображения для модели"""
    image = image.resize((150, 150))
    return image

def plot_probabilities(probs):
    """Визуализация вероятностей"""
    fig, ax = plt.subplots()
    y_pos = np.arange(len(CLASSES))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel('Вероятность')
    ax.set_title('Распределение вероятностей по классам')
    return fig

# Выбор способа загрузки изображения
option = st.radio("Выберите способ:", ["Загрузить изображение", "Нарисовать цветок"])

if option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение цветка...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        # Предобработка и отправка на API
        processed_image = preprocess_image(image)
        
        if st.button("Классифицировать"):
            with st.spinner("Анализируем изображение..."):
                # Конвертируем изображение в байты
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Отправляем на API
                response = requests.post(API_URL, files={"file": img_byte_arr})
                
            if response.status_code == 200:
                result = response.json()
                st.success("Результаты классификации:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Предсказанный класс", result['class'])
                    st.metric("Уверенность", f"{result['confidence']*100:.2f}%")
                
                with col2:
                    st.pyplot(plot_probabilities([result['probabilities'][c] for c in CLASSES]))
            else:
                st.error(f"Ошибка: {response.text}")

else:  # Режим рисования
    st.write("Нарисуйте цветок на холсте ниже:")
    
    # Холст для рисования
    canvas_result = st.canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    if canvas_result.image_data is not None:
        # Конвертируем холст в изображение
        image = Image.fromarray(cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2RGB))
        st.image(image, caption="Ваш рисунок", use_column_width=True)
        
        if st.button("Классифицировать рисунок"):
            with st.spinner("Анализируем рисунок..."):
                # Предобработка
                processed_image = preprocess_image(image)
                
                # Конвертируем в байты
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Отправляем на API
                response = requests.post(API_URL, files={"file": img_byte_arr})
                
            if response.status_code == 200:
                result = response.json()
                st.success("Результаты классификации:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Предсказанный класс", result['class'])
                    st.metric("Уверенность", f"{result['confidence']*100:.2f}%")
                
                with col2:
                    st.pyplot(plot_probabilities([result['probabilities'][c] for c in CLASSES]))
            else:
                st.error(f"Ошибка: {response.text}")

st.markdown("---")
st.info("Приложение использует CNN-модель для классификации 5 видов цветов")
