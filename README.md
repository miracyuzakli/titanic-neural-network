# Titanic Neural Network and Data Analysis

![alt text](images/titanic.jpg)

This repository contains code for exploring and analyzing the Titanic dataset, as well as building a neural network model for prediction. The project focuses on data exploration, visualization, preprocessing, and creating a neural network model using Python.

## Content

- `Titanic_EDA.ipynb`: A Jupyter Notebook for exploring and analyzing the Titanic dataset. It includes basic statistics, survival counts visualization, pairplots, and correlation heatmap.
- `titanic_preprocessing.py`: A Python script for preprocessing the Titanic dataset. It handles data cleaning, feature engineering, and splitting the data into training and testing sets.
- `titanic_neural_network.py`: A Python script for building and training a neural network model using Keras. It uses the preprocessed data to predict passenger survival.

## Usage

1. Clone this repository:

```
git clone https://github.com/miracyuzakli/titanic-neural-network.git
cd titanic-neural-network
```


2. Run the Jupyter Notebook for data analysis:

```
python3 Titanic_EDA.py

```


3. Run the preprocessing script:

```
python titanic_preprocessing.py

```


4. Run the neural network script:

```
python titanic_neural_network.py

```

## Requirements

The project requires the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- keras

You can install the required libraries using the following command:

```
pip install -r requirements.txt

```
---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

