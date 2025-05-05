import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import time
from pyngrok import ngrok, conf

# Настройка страницы
st.set_page_config(page_title="Flower Classifier", layout="wide")
st.title("Классификатор цветов")

if 'ngrok_url' not in st.session_state:
    try:
        ngrok.kill()
        conf.get_default().auth_token = "2wedkprrtp4lebqa1mNPA0o6T51_71JibxY4EQyywFxCuJ4mH"
        public_url = ngrok.connect(8000).public_url
        st.session_state.ngrok_url = public_url
    except:
        st.session_state.ngrok_url = "https://e1c0-34-87-103-23.ngrok-free.app"

API_URL = f"{st.session_state.ngrok_url}/predict"
CLASSES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Проверка доступности API
def check_api():
    try:
        response = requests.get(API_URL.replace('/predict', ''), timeout=5)
        return response.status_code == 200
    except:
        return False

# Функции обработки изображений
def preprocess_image(image):
    image = image.resize((150, 150))
    return image

def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(CLASSES))
    ax.barh(y_pos, probs, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel('Вероятность')
    ax.set_title('Распределение вероятностей')
    return fig

# Интерфейс приложения
st.sidebar.header("Информация")
st.sidebar.write(f"API URL: `{API_URL}`")
st.sidebar.write("Статус:", "Доступен" if check_api() else "Недоступен")

option = st.radio("Выберите способ:", ["Загрузить изображение", "Нарисовать цветок"], horizontal=True)

def process_image(image):
    try:
        start_time = time.time()
        
        # Конвертация в байты
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Отправка на API
        response = requests.post(
            API_URL,
            files={"file": ("image.png", img_bytes, "image/png")},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            # Отображение результатов
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Предсказанный класс", result['class'])
                st.metric("Уверенность", f"{result['confidence']*100:.2f}%")
                st.metric("Время обработки", f"{processing_time:.2f} сек")
            
            with col2:
                st.pyplot(plot_probabilities(
                    [result['probabilities'][c] for c in CLASSES]
                ))
        else:
            st.error(f"Ошибка API: {response.text}")
            
    except Exception as e:
        st.error(f"Ошибка обработки: {str(e)}")

if option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение цветка", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", width=300)
        if st.button("Классифицировать"):
            process_image(image)

else:
    st.write("Нарисуйте цветок ниже:")

    stroke_color = st.color_picker("Выберите цвет кисти", "#FFFFFF")  # выбор цвета
    stroke_width = st.slider("Толщина кисти", 1, 25, 10)  # толщина кисти

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        st.image(image, caption="Ваш рисунок", width=300)
        if st.button("Классифицировать рисунок"):
            process_image(image)

st.markdown("---")
st.info("Приложение использует CNN-модель для классификации 5 видов цветов")
