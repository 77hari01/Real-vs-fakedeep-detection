# 🧠 Real vs Deepfake Image Detection

This project is a **Deep Learning-based Real vs Deepfake Detection System** that classifies whether an image is *real* or *deepfake* using the **XceptionNet** model.  
It is deployed using **Streamlit** for an easy and interactive web interface.

---

## 🚀 Features

- Detects whether an image is **Real** or **Deepfake**
- Trained using an **80:20 dataset split**
- Model architecture: **XceptionNet**
- Achieved **90%+ accuracy**
- Built with **TensorFlow** and **Streamlit**
- Simple user interface for uploading and testing images

---

## 🧩 Tech Stack

- **Python**
- **TensorFlow** – Model training and prediction  
- **Streamlit** – Web application framework  
- **NumPy** – Data processing  
- **Pillow** – Image handling  

---

## 📁 Project Structure

MINI/
│
├── app.py # Streamlit app to run the model
├── model.py # Model architecture and training
├── Dataset/ # Dataset used for training and testing
├── requirements.txt # Dependencies + model/dataset links
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone the repository
bash
git clone https://github.com/77hari01/Real-vs-fakedeep-detection.git
bash

cd Real-vs-fakedeep-detection
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
📦 Model & Dataset Links
Model (.h5 file): Download here

Dataset: (Add your dataset link here if available)

⚠️ Make sure to place the downloaded xceptionnew.weights.h5 file in the same directory as model.py.

▶️ Run the App
To start the Streamlit app:

bash
Copy code
streamlit run app.py
Then open the link shown in your terminal (usually http://localhost:8501/) to access the web interface.

📊 Model Performance
Model Used: XceptionNet

Training Split: 80% training / 20% testing

Accuracy: 90%+

Loss: Low and stable after fine-tuning

✨ Future Enhancements
Add support for video deepfake detection

Improve model with transfer learning or hybrid CNN architectures

Deploy using Docker or Streamlit Cloud

👨‍💻 Author
Hariharan — AI & DS Student
Dhanush — AI & DS Student
Building intelligent models to detect fake content using deep learning ❤️

Thank you

---

Would you like me to make this `README.md` more **attractive with badges (accuracy %, Python version, Streamlit version)** and **a sample image section**?  
I can upgrade it to a **GitHub-ready styled version** if you want.
