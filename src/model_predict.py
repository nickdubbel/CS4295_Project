from joblib import dump, load
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

def model_predict(x_test, y_test, model):
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

    input_folder = "data/interim"

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
