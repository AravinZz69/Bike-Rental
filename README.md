# 🚴 Bike Rental Demand Prediction Web App

A full-stack web application that predicts bike rental demand using machine learning (AdaBoost algorithm) with a beautiful Bootstrap interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)
![ML](https://img.shields.io/badge/ML-AdaBoost-orange)

## 📋 Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)

## ✨ Features

- **Real-time Predictions**: Get instant bike rental demand predictions
- **Interactive UI**: Beautiful, responsive Bootstrap 5 interface
- **Visual Feedback**: Demand level indicators and progress bars
- **Smart Recommendations**: AI-powered suggestions based on predictions
- **Input Validation**: Client and server-side validation
- **Error Handling**: Graceful error messages and recovery
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **High Accuracy**: 99% R² score on test data

## 🛠 Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **Pickle** - Model serialization

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **Bootstrap 5.3** - UI framework
- **JavaScript (ES6)** - Interactivity
- **Font Awesome** - Icons

### Machine Learning
- **Algorithm**: AdaBoost Regressor
- **Performance**: R² = 0.990, MAE = 9.97, RMSE = 3.16
- **Features**: 8 input features (time, weather, users)

## 📁 Project Structure

```
bike_rental_app/
│
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Trained ML models (create this)
│   ├── adaboost_bike_rental_model.pkl
│   ├── xgboost_bike_rental_model.pkl
│   ├── minmax_scaler.pkl
│   └── feature_names.pkl
│
├── templates/                  # HTML templates
│   ├── index.html             # Main prediction page
│   └── about.html             # About page
│
└── static/                     # Static files
    ├── css/
    │   └── style.css          # Custom styles
    └── js/
        └── script.js          # JavaScript functions
```

