# Real vs Deepfake Detection System

A deep learning-based web application that detects whether an image is real or generated using deepfake technology. The system uses Xception architecture trained on an 80/20 split dataset and achieves over 90% accuracy.

## 🎯 Project Overview

This project implements a sophisticated deepfake detection system that can analyze images and classify them as either real or AI-generated. The model is deployed using Streamlit, providing an intuitive web interface for users to upload and analyze images in real-time.

## 🚀 Features

- **High Accuracy Detection**: Achieves 90%+ accuracy on test data
- **Deep Learning Architecture**: Utilizes Xception network for robust feature extraction
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Real-Time Predictions**: Instant classification of uploaded images
- **Pre-trained Model**: Ready-to-use H5 model file included

## 📁 Project Structure

```
.
├── MINI/
│   ├── app.py                 # Main Streamlit application
│   ├── model.py              # Model training script
│   ├── requirements.txt      # Project dependencies
│   └── Dataset/              # Training and testing data
├── OPEN EDITORS/
│   ├── model.py              # Model architecture and training
│   ├── Dataset/              # Image datasets
│   └── requirements.txt      # Package requirements
```

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow 2.18.0** - Deep learning framework
- **Streamlit 1.25.0** - Web application framework
- **NumPy 1.26.0** - Numerical computations
- **Pillow 9.5.0** - Image processing
- **Xception Network** - Pre-trained CNN architecture

## 📦 Installation

### Prerequisites

Make sure you have Python 3.7 or higher installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/77hari01/Real-vs-fakedeep-detection.git)
cd deepfake-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt includes:
```
streamlit==1.25.0
tensorflow==2.18.0
numpy==1.26.0
Pillow==9.5.0
```

### Step 3: Download Required Files

Download the following files from the provided Google Drive links:

1. **Dataset**: [Download Dataset](https://drive.google.com/drive/folders/1hT8UAZf6SCJ0MXkntKx_BBCw30OxJ50k?usp=sharing)
2. **Pre-trained Model**: [Download Dataset](https://drive.google.com/file/d/1tLu_wrOkVFeGk9HhfUwaeAWfsdU3iJcw/view?usp=drive_link)

Extract the dataset to the `Dataset/` directory and place the model file (.h5) in the project root.

## 🎓 Model Training

The model uses the Xception architecture with the following specifications:

- **Dataset Split**: 80% training, 20% testing
- **Architecture**: Xception (Transfer Learning)
- **Training Accuracy**: 90%+
- **Model Format**: H5 (Keras format)

### Training Process

1. The dataset is preprocessed and split into training/testing sets
2. Xception network is used as the base model
3. Custom layers are added for binary classification
4. Model is trained and validated
5. Final model is saved as an H5 file

To train the model from scratch:

```bash
python model.py
```

## 🖥️ Running the Application

### Start the Streamlit App

```bash
cd MINI
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. Open the web interface
2. Upload an image using the file uploader
3. Wait for the model to process the image
4. View the prediction result (Real or Deepfake)
5. Check the confidence score

## 📊 Model Performance

- **Training Accuracy**: 90%+
- **Architecture**: Xception
- **Input Size**: 299x299x3 (standard Xception input)
- **Output**: Binary classification (Real/Deepfake)

## 📝 Dataset Information

The dataset contains:
- Real images from authentic sources
- Deepfake/AI-generated images
- Balanced distribution for fair training
- 80/20 train-test split

- **model Link**: [Google Drive](https://drive.google.com/file/d/1tLu_wrOkVFeGk9HhfUwaeAWfsdU3iJcw/view?usp=drive_link)
- **Dataset Link**: [Google Drive](https://drive.google.com/drive/folders/1hT8UAZf6SCJ0MXkntKx_BBCw30OxJ50k?usp=sharing)

## 🔧 Configuration

Key configuration parameters can be modified in `model.py`:

- Image dimensions
- Batch size
- Number of epochs
- Learning rate
- Model architecture layers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open-source and available for educational purposes.

## 👥 Authors

- Hariharan - AI & DS Student
- Dhanush   - AI & DS Student

## 🙏 Acknowledgments

- Xception architecture by François Chollet
- TensorFlow and Keras teams
- Streamlit for the amazing web framework
- Dataset contributors

## 📞 Contact

For questions or support, please open an issue in the repository.

## 🔮 Future Improvements

- [ ] Add video deepfake detection
- [ ] Implement real-time webcam detection

---

**Note**: Make sure to download the dataset and model files from the provided Google Drive link before running the application.
