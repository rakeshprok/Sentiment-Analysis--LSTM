# Sentiment-Analysis-LSTM-IMDb-movie-review

This project focuses on sentiment analysis of IMDb movie reviews, leveraging deep learning techniques and word embeddings. The goal is to predict the sentiment (positive or negative) of a movie review based on its textual content. Using Python and deep learning libraries, I built a system that analyzes the sentiment of reviews, providing insights into how viewers react to movies.

# Project Overview
In this project, I explored the IMDb movie review dataset, which contains thousands of reviews labeled with their sentiment. The project involves multiple steps, including data preprocessing, building and training machine learning models, and using them to classify reviews. Below is a breakdown of the key components:

# Steps Involved:
# 1. Data Collection and Preprocessing
Dataset: The dataset comprises a large volume of IMDb movie reviews, each labeled as either positive or negative.

Text Preprocessing: The text data underwent rigorous preprocessing, including the removal of HTML tags, punctuation, and special characters, followed by tokenization and padding to standardize input lengths for model training.

# 2. Word Embeddings
GloVe Embeddings: Pre-trained GloVe word embeddings were employed to represent words as dense vectors, capturing rich semantic relationships crucial for improving model performance.

# 3. Model Development
The project leverages state-of-the-art deep learning model to perform sentiment classification:

Long Short-Term Memory (LSTM): Implemented to capture temporal dependencies and contextual information across the sequence of words, making it ideal for text data with sequential patterns like movie reviews.

# 4. Model Training and Evaluation

Training: Models were trained using the preprocessed text data, with optimization techniques applied to enhance convergence and avoid overfitting. Advanced configurations of hyperparameters were used to fine-tune model performance.

Evaluation: The models were evaluated based on key metrics such as accuracy, precision, recall, and F1-score. The LSTM model demonstrated superior performance due to its ability to effectively model sequential dependencies and context within the reviews.

# 5. Real-Time Sentiment Prediction
I developed a predictive function that allows users to input a movie review and get a real-time sentiment prediction (positive or negative) based on the trained LSTM model.

# Tools and Technologies Used
Python: For data processing, model building, and evaluation.

Pandas & NumPy: For handling and processing the dataset.

Kaggle API: To retrieve the IMDb movie review dataset.

GloVe: Pre-trained word embeddings for representing words in the text data.

TensorFlow & Keras: For building and training the deep learning models.

Matplotlib & Seaborn: For data visualization and performance plots.

# Key Results
The LSTM model achieved good accuracy and robustness in sentiment classification, demonstrating strong performance in understanding the sentiment of movie reviews by capturing long-term dependencies.

# Key Takeaways
This project showcases how deep learning techniques such as LSTMs can be effectively applied to natural language processing tasks like sentiment analysis.

Effective Use of Word Embeddings: GloVe embeddings played a crucial role in improving model accuracy by capturing the semantic relationships between words.


Feel free to explore the code and notebooks for a comprehensive understanding of the workflow, model architecture, and evaluation process!
