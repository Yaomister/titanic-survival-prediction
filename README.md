# Titanic Survival Prediction (NumPy Only)

Personal playground project for revisiting the classic Kaggle Titanic dataset and brushing up on the fundamentals. Everything lives in `titanic_data_prediction.ipynb`, where I explore the data, clean it up, and build a logistic regression model from scratch using only NumPy.

## What's inside

- Quick exploratory analysis of the `titanic.csv` dataset (null checks, class and sex survival rates, etc.).
- Lightweight preprocessing: drop text-heavy columns, fill `Age`/`Embarked` gaps, and one-hot encode the embarkation port.
- Manual implementation of sigmoid activation, binary cross-entropy loss, and gradient descent for a baseline classifier.
- Simple loss curve visualization plus a training accuracy printout (~80% with the current setup).

## How to use it

1. Open the notebook: `titanic_data_prediction.ipynb`.
2. Run the cells top to bottom (Python 3 with NumPy, pandas, and Matplotlib installed).
3. Tweak the feature list, learning rate, or epochs to experiment with different behaviors.

## Ideas to extend

- Hold out a validation split or add cross-validation for a fairer accuracy number.
- Engineer interaction features (family size, fare per person, titles extracted from names, etc.).
- Compare the NumPy baseline against scikit-learn or PyTorch models for fun.
