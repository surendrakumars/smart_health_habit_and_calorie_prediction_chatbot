# Smart Health Habit and Calorie Prediction Chatbot

## Overview

The Smart Health Habit and Calorie Prediction Chatbot is an intelligent web-based chatbot that analyzes a user's daily lifestyle habits through natural language conversation and predicts the number of calories burned based on the extracted features.

The chatbot interacts with users conversationally to collect health-related information such as sleep duration, physical activity, mood, and daily habits. Using this information, the system processes the extracted features and applies a machine learning model to predict calorie expenditure.

This project demonstrates the integration of Natural Language Processing, Machine Learning, and Web Development to create an interactive health analysis system.

---

## Key Features

- Conversational chatbot interface
- Natural language input processing
- Feature extraction from user responses
- Calorie prediction using machine learning
- Health habit analysis
- Flask-based backend
- Interactive web interface

---

## Technologies Used

### Programming Language
- Python

### Framework
- Flask

### Libraries
- Pandas
- NumPy
- Scikit-learn
- JSON
- Regex / Basic NLP techniques

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- Random Forest Regression Model (for calorie prediction)

---

## System Workflow

1. The chatbot initiates a conversation with the user.
2. The user provides health-related information through natural language.
3. The system processes the text input using basic NLP techniques.
4. Important features such as steps, sleep hours, mood, and activity are extracted.
5. These features are passed to a trained machine learning model.
6. The model predicts the estimated calories burned.
7. The predicted result is displayed to the user.

---

## Dataset

The model was trained using a dataset containing lifestyle and activity information.

### Dataset Features

- Date
- Step Count
- Mood
- Calories Burned
- Hours of Sleep
- Activity Status
- Weight (kg)

The dataset helps the model learn patterns between daily habits and calorie expenditure.

---

## Project Structure

```
smart_health_habit_and_calorie_prediction_chatbot
│
├── app.py
├── model.pkl
├── dataset.csv
│
├── templates
│   └── index.html
│
├── static
│   ├── style.css
│   └── script.js
│
├── utils
│   └── feature_extraction.py
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/smart_health_habit_and_calorie_prediction_chatbot.git
```

Move into the project directory

```bash
cd smart_health_habit_and_calorie_prediction_chatbot
```

Install required dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask server

```bash
python app.py
```

Open your browser and go to

```
http://127.0.0.1:5000
```

The chatbot interface will appear and you can start interacting with it.

---

## Example Interaction

User:
```
I slept for 7 hours and walked around 6000 steps today.
```

Chatbot:
```
Based on your daily habits and activity level, your estimated calories burned today are approximately 2100 calories.
```

---

## Applications

- Personal health monitoring
- Fitness tracking systems
- AI-based health assistants
- Lifestyle analysis tools
- Smart wellness applications

---

## Future Improvements

- Integration with wearable fitness devices
- Advanced NLP models for better conversation understanding
- Personalized health recommendations
- Mobile application integration
- Real-time activity tracking

---

## Learning Outcomes

This project helped in understanding and implementing:

- Natural Language Processing basics
- Machine learning model integration
- Feature extraction from conversational input
- Flask backend development
- Frontend and backend integration
- AI-powered web application development

---

## Author

Surendra Kumar S  
B.Tech Computer Science and Engineering
