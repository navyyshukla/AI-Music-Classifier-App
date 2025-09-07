# AI Music Genre Classifier 🎵

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-music-classifier.streamlit.app/##ai-music-genre-classifier)

**[➡️ Click Here to View the Live Application](https://ai-music-classifier.streamlit.app/##ai-music-genre-classifier)**

This project is an end-to-end deep learning application that automatically classifies the genre of a piece of music from a raw audio file. It leverages advanced audio signal processing and computer vision techniques to achieve high accuracy on the GTZAN dataset.

The entire pipeline, from data processing to a fully interactive web application, has been built and deployed on Streamlit Community Cloud.

## Key Features

* **Deep Learning with CNNs:** Utilizes Convolutional Neural Networks (CNNs), the state-of-the-art for image-based pattern recognition, to classify music genres with high accuracy.
* **Audio to Image Conversion:** Transforms raw audio signals into Mel Spectrograms, converting the audio classification problem into a more powerful image classification task.
* **Advanced Data Augmentation:** The training dataset is automatically augmented (noise injection, pitch shifting, time stretching) to create a larger, more robust dataset, significantly improving the model's ability to generalize.
* **Multi-Model Comparison:** The training pipeline builds and evaluates three different CNN architectures (including a powerful transfer learning model with MobileNetV2) and automatically selects the best-performing one for deployment.
* **Interactive Web Application:** A beautiful and user-friendly web interface built with Streamlit that allows users to upload their own audio files and receive real-time predictions and visualizations.
* **End-to-End MLOps:** The project covers the complete machine learning lifecycle, from data preprocessing and model training to version control with Git and final deployment on a cloud platform.

---

## Tech Stack

* **Core Language:** Python
* **Data Processing & Machine Learning:**
    * **TensorFlow / Keras:** For building, training, and evaluating the CNN models.
    * **Librosa:** For advanced audio signal processing and Mel Spectrogram generation.
    * **Scikit-learn:** For performance metrics and data splitting.
    * **NumPy & Pandas:** For numerical data manipulation.
* **Web Application & Visualization:**
    * **Streamlit:** For creating and serving the interactive web application.
    * **Plotly & Matplotlib:** For generating dynamic charts and spectrogram images.
* **Deployment & Version Control:**
    * **Streamlit Community Cloud:** For hosting the live application.
    * **GitHub:** For version control and as the deployment source.
    * **Git LFS (Large File Storage):** For handling the large trained model file.

---

## How to Run This Project Locally

To set up and run this project on your own machine, follow these steps.

### Prerequisites
* Git and Git LFS installed.
* Python 3.10+ installed.

### Step-by-Step Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/navyyshukla/AI-Music-Classifier-App](https://github.com/navyyshukla/AI-Music-Classifier-App)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install all required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Data:** This repository does not include the raw audio data. Please download the **GTZAN Genre Collection** dataset and ensure the `genres_original` folder is placed inside a `Data` folder in the project root.

5.  **Run the Feature Extraction Script:** This will create the necessary spectrogram images. This will take 15-20 minutes.
    ```bash
    python feature_extractor.py
    ```

6.  **Run the Model Training Script:** This will train the CNN models and save the best one. This can take a long time (30 minutes to over an hour).
    ```bash
    python train_model.py
    ```

7.  **Run the Streamlit Application:** Once the model is trained, you can launch the web app.
    ```bash
    streamlit run app.py
    ```

The application will now be running locally in your web browser!
