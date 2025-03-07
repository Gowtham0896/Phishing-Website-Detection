# 🔍 Phishing Website Detection using Machine Learning  

## 📌 Project Overview  
This project focuses on detecting phishing websites using machine learning and deep learning techniques. It leverages **TensorFlow, Scikit-Learn, and NLP-based feature extraction** to classify URLs as phishing or legitimate. A **Streamlit web application** is integrated for real-time phishing detection.  

## 🚀 Features  
✅ Machine learning-based phishing website classification  
✅ NLP techniques for feature extraction from URLs  
✅ Model training with multiple classifiers (SVM, Random Forest, Gradient Boosting, etc.)  
✅ Hyperparameter tuning to improve accuracy  
✅ Deployment using **Streamlit** for real-time URL verification  

## 🛠️ Technologies & Tools  
- **Programming Language:** Python  
- **Machine Learning Frameworks:** TensorFlow, Scikit-Learn  
- **Data Processing:** Pandas, NumPy, SciPy  
- **NLP & Feature Extraction:** NLTK, BeautifulSoup, CountVectorizer  
- **Algorithms Used:** Logistic Regression, Naïve Bayes, SVM, Random Forest, Gradient Boosting, Decision Tree, KNN  
- **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV  
- **Evaluation Metrics:** Accuracy, F1 Score, Precision, Recall, Confusion Matrix  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  

## 📂 Project Structure  
```
├── data/                     # Dataset (phishing URLs)
├── notebooks/                 # Jupyter Notebooks for EDA and Model Training
├── models/                    # Saved ML models
├── streamlit-app.py           # Streamlit application for phishing detection
├── Incognito_ML_Project.ipynb # Machine learning model implementation
├── requirements.txt           # Dependencies and libraries
├── README.md                  # Project documentation (this file)
```

## 📊 Dataset  
The dataset used for training was sourced from **Kaggle**:  
🔗 [Phishing Site URLs Dataset](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)  

## 🖥️ Installation & Setup  
To run this project locally, follow these steps:  

### 1️⃣ Clone the Repository  
```bash
https://github.com/Gowtham0896/Phishing-Website-Detection.git
cd phishing-detection
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Web App  
```bash
streamlit run streamlit-app.py
```

## 🏆 Results  
- Processed **500,000+ phishing URLs** with NLP-based feature extraction  
- Achieved **98.5% accuracy** using ensemble learning  
- Improved model performance through **hyperparameter tuning**  

## 📌 Future Enhancements  
🔹 Implement **deep learning** techniques for better phishing detection  
🔹 Integrate **real-time WHOIS API** for additional website verification  
🔹 Expand dataset with newer phishing domains  

## 📜 License  
This project is open-source and available under the **MIT License**.  

## 🤝 Contributing  
Contributions are welcome! Feel free to submit a pull request.  

---
