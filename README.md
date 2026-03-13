# 🏥 Smart Health Habit and Calorie Prediction Chatbot

An intelligent conversational AI chatbot designed to help users track their health habits, predict calorie intake, and receive personalized health recommendations. Built with machine learning and natural language processing, this chatbot serves as your personal health assistant.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## ✨ Features

### Core Functionality
- **Intelligent Chatbot Interface**: Natural language conversation for health inquiries and guidance
- **Calorie Prediction**: ML-based predictions for food items and meals
- **Health Habit Tracking**: Monitor daily health behaviors and patterns
- **Personalized Recommendations**: AI-driven suggestions based on user profile and history
- **Interactive Dashboard**: Web-based interface for visualization and management

### Advanced Capabilities
- Real-time calorie calculations
- Multi-meal tracking and daily summaries
- Health metrics analysis
- User preference learning
- Historical data visualization
- Nutritional information database integration

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API and web interface
- **TensorFlow/Keras**: Machine learning models
- **Scikit-learn**: Data preprocessing and model training
- **NLTK/SpaCy**: Natural language processing
- **SQLite/PostgreSQL**: Data persistence

### Frontend
- **HTML5**: Page structure
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive elements and API calls
- **Bootstrap/Tailwind**: UI framework (optional)

### Machine Learning
- Calorie prediction model (trained on nutritional datasets)
- Health habit classification model
- Intent recognition model for chatbot

### Tools & Libraries
- Pandas: Data manipulation
- NumPy: Numerical computations
- Matplotlib/Plotly: Data visualization
- Requests: HTTP client library

---

## 📁 Project Structure

```
smart_health_habit_and_calorie_prediction_chatbot/
│
├── src/                           # Source code
│   ├── app.py                     # Main Flask application
│   ├── chatbot.py                 # Chatbot logic and NLP processing
│   ├── models/                    # ML model implementations
│   │   ├── calorie_predictor.py   # Calorie prediction model
│   │   └── habit_classifier.py    # Health habit classification
│   ├── utils/                     # Utility functions
│   │   ├── data_processor.py      # Data processing utilities
│   │   └── helpers.py             # Helper functions
│   └── database/                  # Database operations
│       └── db_handler.py          # Database CRUD operations
│
├── models/                        # Pre-trained models
│   ├── calorie_model.pkl          # Saved calorie prediction model
│   ├── intent_classifier.pkl      # Chatbot intent recognition
│   └── nlp_model.bin              # NLP model files
│
├── data/                          # Data files
│   ├── training_data/             # Training datasets
│   ├── food_database.csv          # Food items and nutrition info
│   └── user_habits.csv            # Sample user health data
│
├── templates/                     # HTML templates
│   ├── index.html                 # Main chat interface
│   ├── dashboard.html             # User dashboard
│   ├── base.html                  # Base template
│   └── components/                # Reusable components
│
├── static/                        # Static assets
│   ├── css/                       # Stylesheets
│   │   └── style.css
│   ├── js/                        # JavaScript files
│   │   ├── chat.js
│   │   └── dashboard.js
│   └── images/                    # Image assets
│
├── docs/                          # Documentation
│   ├── API_DOCS.md                # API endpoints documentation
│   ├── MODEL_DOCS.md              # Model architecture and training
│   └── SETUP_GUIDE.md             # Detailed setup instructions
│
├── tests/                         # Unit and integration tests
│   ├── test_chatbot.py
│   ├── test_models.py
│   └── test_api.py
│
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration settings
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/surendrakumars/smart_health_habit_and_calorie_prediction_chatbot.git
cd smart_health_habit_and_calorie_prediction_chatbot
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Example .env file:
# FLASK_ENV=development
# SECRET_KEY=your-secret-key-here
# DATABASE_URL=sqlite:///health_chatbot.db
# API_KEY=your-api-key
```

### Step 5: Initialize Database

```bash
python src/database/db_handler.py --init
```

### Step 6: Run the Application

```bash
# Development mode
python src/app.py

# Production mode
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

The application will be available at `http://localhost:5000`

---

## ⚙️ Configuration

### Config File (`config.py`)

```python
# Example configuration
class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'your-secret-key'
    DATABASE_URL = 'sqlite:///health_chatbot.db'
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
```

### Environment Variables

Key environment variables:
- `FLASK_ENV`: `development` or `production`
- `SECRET_KEY`: Secret key for session management
- `DATABASE_URL`: Database connection string
- `API_KEY`: Optional API key for external services
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

---

## 📖 Usage

### Starting the Chatbot

1. **Access the Web Interface**
   - Open your browser and navigate to `http://localhost:5000`
   - You'll see the chat interface

2. **Interact with the Chatbot**
   - Ask about food items: *"What's the calorie count for an apple?"*
   - Log meals: *"I ate a chicken sandwich and orange juice"*
   - Track habits: *"Log my 30-minute run"*
   - Get recommendations: *"What should I eat for breakfast?"*

### API Usage

#### Get Calorie Prediction
```bash
curl -X POST http://localhost:5000/api/predict-calories \
  -H "Content-Type: application/json" \
  -d '{"food_item": "grilled chicken breast", "quantity": 200}'
```

#### Log User Activity
```bash
curl -X POST http://localhost:5000/api/log-activity \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "activity": "running", "duration": 30, "calories_burned": 300}'
```

#### Get User Summary
```bash
curl -X GET http://localhost:5000/api/user-summary/1
```

---

## 📡 API Documentation

### Endpoints

#### Chatbot Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to chatbot |
| GET | `/api/chat/history/<user_id>` | Get chat history |

#### Calorie Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict-calories` | Predict calories for food item |
| GET | `/api/food-database` | Get food database |
| POST | `/api/log-meal` | Log a meal |

#### Health Tracking Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/log-activity` | Log physical activity |
| GET | `/api/user-summary/<user_id>` | Get user health summary |
| GET | `/api/daily-report/<user_id>` | Get daily health report |

For detailed API documentation, see [API_DOCS.md](docs/API_DOCS.md)

---

## 🤖 Model Details

### Calorie Prediction Model
- **Type**: Regression model
- **Algorithm**: XGBoost / Random Forest
- **Input Features**: Food item, quantity, cooking method, ingredients
- **Output**: Estimated calorie count
- **Accuracy**: ~92%

### Intent Classification Model
- **Type**: Multi-class classification
- **Algorithm**: LSTM / Transformer-based
- **Input**: User message text
- **Output**: Intent category (calorie_query, habit_logging, etc.)

### Health Habit Classifier
- **Type**: Sequential pattern recognition
- **Algorithm**: Hidden Markov Model
- **Input**: User activity logs
- **Output**: Health habit classification and recommendations

For detailed model information, see [MODEL_DOCS.md](docs/MODEL_DOCS.md)

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_chatbot.py

# Run with coverage
pytest --cov=src tests/
```

---

## 📊 Data Format

### User Input Example
```json
{
  "user_id": 1,
  "message": "I had a grilled chicken sandwich for lunch",
  "timestamp": "2024-03-13T12:30:00Z"
}
```

### Chatbot Response Example
```json
{
  "response": "Great! A grilled chicken sandwich typically contains around 350-450 calories. Would you like me to log this meal?",
  "intent": "calorie_query",
  "food_items": ["grilled chicken sandwich"],
  "estimated_calories": 400,
  "timestamp": "2024-03-13T12:30:05Z"
}
```

---

## 🔧 Development

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature** following the project structure

3. **Add tests** in the `tests/` directory

4. **Submit a pull request**

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions small and focused

---

## 📦 Dependencies

Key dependencies (see `requirements.txt` for complete list):
- Flask>=2.0.0
- TensorFlow>=2.8.0
- Scikit-learn>=1.0.0
- NLTK>=3.6.0
- Pandas>=1.3.0
- NumPy>=1.21.0
- SQLAlchemy>=1.4.0

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- New features include appropriate tests
- Documentation is updated

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🆘 Support

### Getting Help

- **Documentation**: Check [docs/](docs/) folder for detailed guides
- **Issues**: Report bugs on [GitHub Issues](https://github.com/surendrakumars/smart_health_habit_and_calorie_prediction_chatbot/issues)
- **Discussion**: Use [GitHub Discussions](https://github.com/surendrakumars/smart_health_habit_and_calorie_prediction_chatbot/discussions)

### Troubleshooting

#### Issue: ModuleNotFoundError
```bash
# Solution: Ensure dependencies are installed
pip install -r requirements.txt
```

#### Issue: Database Connection Error
```bash
# Solution: Check DATABASE_URL in .env file
# Verify database file exists or reinitialize
python src/database/db_handler.py --init
```

#### Issue: Port Already in Use
```bash
# Solution: Run on different port
python src/app.py --port 8000
```

---

## 🗺️ Roadmap

- [ ] Mobile app integration
- [ ] Advanced meal planning features
- [ ] Integration with fitness trackers (Fitbit, Apple Health)
- [ ] Multi-language support
- [ ] Enhanced nutritional recommendations
- [ ] Community features and challenges
- [ ] API rate limiting and authentication

---

## 📞 Contact

**Author**: Surendra Kumar S  
**Email**: surendrakumars@example.com  
**GitHub**: [@surendrakumars](https://github.com/surendrakumars)

---

## 🙏 Acknowledgments

- Special thanks to the open-source community
- Nutritional data sourced from public databases
- ML algorithms inspired by research papers and best practices

---

**Last Updated**: March 2024  
**Version**: 1.0.0

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [NLTK Book](https://www.nltk.org/book/)

---

**⭐ If you find this project helpful, please consider giving it a star!**
