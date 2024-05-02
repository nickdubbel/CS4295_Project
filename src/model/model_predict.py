"""This module uses the trained model to make predictions and evaluates the model's performance."""

import os
from joblib import dump, load

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np
np.random.seed(0)

def model_predict(x_test, y_test, model):
    """Makes predictions using the trained model and determines model accuracy."""
    # predict
    y_pred = model.predict(x_test, batch_size=1000)
    print(y_pred)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:',accuracy_score(y_test,y_pred_binary))

    return report, confusion_mat

def main():
    """Makes a prediction and stores results in a folder."""
    input_folder = "data/interim"

    # check if model and load data exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' is empty")

    # Load model
    model = load(f'{input_folder}/model.joblib')

    # Load data
    x_test = load(f'{input_folder}/x_data.joblib')[2]
    y_test = load(f'{input_folder}/y_data.joblib')[2]

    # Predict
    report, confusion_mat = model_predict(x_test, y_test, model)

    output_folder = "data/interim"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dump(report, f'{output_folder}/report.joblib')
    dump(confusion_mat, f'{output_folder}/confusion_mat.joblib')
    print("Report and Confusion Matrix saved at output folder")


if __name__ == "__main__":
    main()
