from joblib import dump, load
import os

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout


def model_train(x_train, y_train, x_val, y_val, char_index):
   

    params = {'loss_function': 'binary_crossentropy',
                        'optimizer': 'adam',
                        'sequence_length': 200,
                        'batch_train': 5000,
                        'batch_test': 5000,
                        'categories': ['phishing', 'legitimate'],
                        'char_index': None,
                        'epoch': 30,
                        'embedding_dimension': 50}



    model = Sequential()
    voc_size = len(char_index.keys())
    print("voc_size: {}".format(voc_size))
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))



    #compile the model
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])


    hist = model.fit(x_train, y_train,
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(x_val, y_val)
                    )
    
    return model, hist

#main function
def main():
    
    input_folder = "data/interim"

    # check if there is data inside the folder
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' is empty")
    
    # load data
    char_index = load(f'{input_folder}/char_index.joblib')
    x = load(f'{input_folder}/x_data.joblib')
    y = load(f'{input_folder}/y_data.joblib')

    # train model
    model, hist = model_train(x[0], y[0], x[1], y[1], char_index)
    
    output_folder = "data/interim"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dump(model, f'{output_folder}/model.joblib')
    dump(hist, f'{output_folder}/hist.joblib')
    print("Model saved at output folder")

if __name__ == "__main__":
    main()