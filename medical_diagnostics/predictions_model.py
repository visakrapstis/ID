import tensorflow as tf
import pandas as pd
import numpy as np


model = tf.keras.models.load_model("134_disease_prediction_model.keras")

all_symptoms = pd.read_csv("all_symptoms.csv", index_col=0)["0"].unique().tolist()
all_diseases = pd.read_csv("labels.csv", index_col=0)["disease"].unique().tolist()


#   the functions already created in modeling notebook, but here also optimized for a good functionality in streamlit environment

def get_all_data(disease):
    dataframe = pd.read_csv("ml_main.csv")
    all_patients = []
    disease_symptoms = set()

    disease_df = dataframe[dataframe["disease"]==disease]
    for patient in disease_df.iloc:
        for index, symptom in enumerate(patient.values):
            if symptom == 1.0:
                all_patients.append(disease_df.columns[index])
                disease_symptoms.add(disease_df.columns[index])
    disease_dict = {}
    for symptom in disease_symptoms:
        frequency = int((100 * all_patients.count(symptom)) / len(disease_df))
        disease_dict.update({symptom: frequency})
    sorted_dict = dict(sorted(disease_dict.items(), key=lambda x: x[1], reverse=True))
    return f"\nfor a disease: {disease}\n\nout of {len(disease_df)} cases\n\nall symptoms and frequencies: \n\n", sorted_dict


def full_answer(human_guess):

    def change_zeros_to_ones(symptoms):
        guess = np.zeros((1, 396))
        indexes = []
        for symptom in symptoms:
            indexes.append(all_symptoms.index(symptom))
        for col in indexes:
            guess[0][col] = 1
        return guess

    guessed_diseases_list = model.predict(change_zeros_to_ones(human_guess)).tolist()[0]
    name_the_top = {}
    for index, i in enumerate(sorted(guessed_diseases_list,reverse=True)[:5]):
        disease_code = guessed_diseases_list.index(i)
        name_the_top.update({all_diseases[disease_code]: f"{int(i*100)}%"})
    return name_the_top