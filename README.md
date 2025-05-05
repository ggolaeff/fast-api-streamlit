# 🌸 Flower Image Classifier

## 📌 Описание проекта

Данный проект посвящён **классификации изображений цветов** по пяти типам: **Lilly, Lotus, Sunflower, Orchid и Tulip**. С применением различных архитектур нейронных сетей мы сравниваем точность, скорость и эффективность предсказаний.

### 📂 Используемый датасет

[5 Flower Types Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)

- **Классы**: Lilly, Lotus, Sunflower, Orchid, Tulip  
- **Объём**: 1000 изображений (по 1000 на каждый класс)  
- **Формат**: Цветные изображения RGB  

---

## 🧠 Сравниваемые модели

| № | Модель              | Описание |
|--:|---------------------|----------|
| 1 | Basic CNN           | Простая сверточная сеть с 3 Conv-блоками, Dropout и Dense |
| 2 | ResNet-Inspired     | Пользовательская модель с остаточными блоками |
| 3 | MLP (Dense-only)    | Полносвязная модель без сверточных слоёв |
| 4 | ResNet50 + Transfer | Предобученная ResNet50 с замороженной базой и кастомной "головой" |

---

## 📊 Таблица сравнения моделей

| Модель               | Accuracy | Precision | Recall | F1    | Inference Time (сек) |
|----------------------|----------|-----------|--------|-------|-----------------------|
| **Basic CNN**        | 0.826    | 0.824     | 0.826  | 0.824 | 7.13                  |
| **ResNet Inspired**  | 0.679    | 0.683     | 0.679  | 0.651 | 9.12                  |
| **MLP Classifier**   | 0.417    | 0.479     | 0.417  | 0.377 | 2.70                  |
| **ResNet50 Transfer**| 0.200    | 0.040     | 0.200  | 0.067 | 23.09                 |

---

## 📈 Визуализации результатов

![image](https://github.com/user-attachments/assets/f13e9d10-3fdf-47b5-8e72-9fc116cb91c6)
![image](https://github.com/user-attachments/assets/692eece8-61b2-43b9-a992-acd456cd9e65)


---

## ⚙️ Инструкция по локальному запуску

1. **Клонируйте репозиторий**

```bash
git clone https://github.com/your-username/flower-classification.git
cd flower-classification
```

2. **Установите зависимости**

```bash
pip install -r requirements.txt
```

3. **Запустите FastAPI**

```bash
uvicorn main:app --reload
```

4. **Запустите Streamlit-приложение**

```bash
streamlit run app.py
```

---

## 🌐 Развёрнутые сервисы

-  **FastAPI API**: [https://e1c0-34-87-103-23.ngrok-free.app](https://e1c0-34-87-103-23.ngrok-free.app)
-  **Streamlit App**: [https://fast-api-app-cfgmtyobdmrv5mxysdwzdz.streamlit.app/](https://fast-api-app-cfgmtyobdmrv5mxysdwzdz.streamlit.app/)

---

## 📬 Пример использования API

```bash
curl -X POST "https://e1c0-34-87-103-23.ngrok-free.app/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_image.jpg"
```

Пример ответа:

```json
{
  "class": "Sunflower",
  "confidence": 0.9821,
  "probabilities": {
    "Lilly": 0.001,
    "Lotus": 0.004,
    "Orchid": 0.006,
    "Sunflower": 0.982,
    "Tulip": 0.007
  },
  "processing_time": "0.324 сек"
}
```


