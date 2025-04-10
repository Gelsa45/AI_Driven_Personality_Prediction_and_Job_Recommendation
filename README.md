# 🧠 AI-Driven Personality Prediction and Career Recommendation System

This project predicts MBTI personality traits (with a focus on **Extraversion**) from user-input text, ensures data authenticity by detecting AI-generated content, and recommends suitable careers based on the predicted personality using machine learning models.

## 🚀 Features

- 🔍 **Personality Trait Prediction** using BiLSTM and NLP techniques
- 🤖 **AI-Generated Text Detection** to ensure input authenticity
- 🧑‍💼 **Career Recommendation System** using Random Forest classifier
- 📊 Real-time and user-friendly interface for interaction and output

---

## 📌 Project Workflow

1. **User Input**: User submits a piece of text.
2. **AI Detection**: System checks if the input is AI-generated.
3. **Personality Prediction**: Predicts MBTI traits using BiLSTM.
4. **Job Recommendation**: Suggests jobs based on the predicted trait.
5. **Result Display**: Shows personality, job suggestions, and AI detection status.

---

## 🧠 Models Used

| Model | Purpose | Description |
|-------|---------|-------------|
| **BiLSTM** | Personality Prediction | Captures forward and backward text context for accurate personality classification |
| **Random Forest** | Career Recommendation | Maps personality traits to job roles using labeled dataset |
| **AI Text Detection Model** | Input Validation | Detects whether the text is AI-generated to ensure data integrity |

---
## 🧪 Technologies Used

- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- GloVe embeddings
- HTML/CSS (Frontend)

---

## 📈 Results

| Model       | Accuracy | Training Time | Input Dependency |
|-------------|----------|----------------|------------------|
| SVM         | ~75%     | Fast (~5 min)  | Requires manual features |
| BiLSTM      | ~84%     | Slow (~30 min) | Learns features automatically |

---

## 🌐 Use Cases

- 🎯 Career Counseling
- 🏢 HR & Recruitment
- 👥 Team Building
- 📱 Social Media Analysis

---

## 🔮 Future Scope

- 🌍 Multilingual Text Support (e.g., Hindi, Malayalam)
- ⚡ Real-time Personality Detection
- 🤖 Integration of advanced models like GPT-4, XLNet

---

## 📄 References

1. [Getting Personal: A Deep Learning Artifact... (2023)](https://ieeexplore.ieee.org/document/10735203)  
2. [Extrovert and Introvert Classification using SVM (2020)](https://www.researchgate.net/publication/340399695)  
3. [SVM Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)  
4. [LSTM Networks Explained - GeeksforGeeks](https://www.geeksforgeeks.org/understanding-of-lstm-networks/)  
5. [MBTI-Based Recommendation Paper (2022)](https://www.researchgate.net/publication/358396484)





