# Salary Prediction Model

This project implements a machine learning model for predicting salaries based on features extracted from hh.ru resume data. It provides a command-line interface for training models and making predictions.

## Project Structure

- `app.py` - Main CLI application with predict and train commands
- `model.py` - SalaryModel class implementing the prediction logic
- `config.py` - Configuration settings and paths
- `utils.py` - Helper functions for data loading, preprocessing, and model persistence
- `requirements.txt` - Project dependencies

## Features

- **Random Forest Regressor** for salary prediction
- **Feature scaling** using StandardScaler
- **Model persistence** with joblib serialization
- **Train/test split** for model evaluation
- **NaN handling** and data validation
- **Comprehensive logging** for debugging

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The application supports two main commands:

### Training a Model

Train a new model on your data:

```bash
python app.py train path/to/x_data.npy path/to/y_data.npy
```

This will:
- Load the training data from `.npy` files
- Split data into training and testing sets
- Scale features using StandardScaler
- Train a Random Forest Regressor
- Save the model and scaler to the `resources/` directory
- Display R² and MSE metrics

### Making Predictions

Use a trained model to predict salaries:

```bash
python app.py predict path/to/x_data.npy
```

This will:
- Load the trained model and scaler from `resources/`
- Process the input features
- Output salary predictions (one per line)

## Configuration

The `config.py` file contains all configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BASE_DIR` | Project root directory | Auto-detected |
| `PARSING_DIR` | Directory with training data | `../parsing/` |
| `RESOURCES_DIR` | Directory for saved models | `./resources/` |
| `TRAIN_X_FILE` | Default training features path | `../parsing/x_data.npy` |
| `TRAIN_Y_FILE` | Default training targets path | `../parsing/y_data.npy` |
| `MODEL_FILENAME` | Saved model filename | `trained_model.joblib` |
| `SCALER_FILENAME` | Saved scaler filename | `scaler.joblib` |
| `TEST_SIZE` | Proportion for test split | 0.2 |
| `RANDOM_STATE` | Random seed for reproducibility | 42 |

## Model Details

The project uses a **Random Forest Regressor** with the following characteristics:
- 100 estimators (trees)
- Parallel processing (`n_jobs=-1`)
- Features are standardized using StandardScaler
- Automatic handling of NaN values (removal during training, mean imputation during prediction)

## Input Data Format

### Training Data
- `X.npy`: Feature matrix with shape `(n_samples, n_features)`
- `y.npy`: Target values (salaries) with shape `(n_samples,)`

### Prediction Data
- `X.npy`: Feature matrix with shape `(n_samples, n_features)`

## Output Format

Predictions are output as one salary value per line, rounded to 2 decimal places (rubles and kopecks).

## Logging and Debugging

The `utils.py` module provides comprehensive logging:

- Data loading status and shapes
- NaN value detection and handling
- Training progress and metrics
- Model saving/loading confirmation

## Error Handling

The application includes robust error handling:
- File existence validation
- Extension checking (.npy files only)
- Data shape compatibility verification
- NaN and infinite value detection
- Graceful error messages with emoji indicators

## Program Output Examples

### Training
```
🧠 Обучение модели на данных:
   X: ../parsing/x_data.npy
   Y: ../parsing/y_data.npy
🗑️  Удаляю предыдущую модель...
📥 Загрузка данных для обучения:
   X: ../parsing/x_data.npy
   Y: ../parsing/y_data.npy
📊 Данные для обучения: 1500 образцов, 8 признаков
🧠 Обучение RandomForestRegressor...
📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
   R²: 0.85
   MSE: 1250000.00
💾 Модель сохранена в ./resources
✅ Модель успешно обучена и сохранена
```

### Prediction
```
📥 Загружено 500 образцов для предсказания
✅ Сделано 500 предсказаний
45000.00
52000.00
38500.00
...
```

## Dependencies

- Python 3.6+
- pandas
- numpy
- scikit-learn
- joblib

## Notes

- The model automatically creates the `resources/` directory if it doesn't exist
- Previous models are overwritten when training with new data
- During prediction, missing values (NaN) are replaced with column means
- The model uses mean imputation only for prediction; training data with NaN is filtered out