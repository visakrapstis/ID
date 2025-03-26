import tensorflow as tf
from predictions_model import full_answer, all_symptoms, all_diseases, get_all_data
import streamlit as st


symptoms_list = all_symptoms

st.title("Symptom Checker")
st.write("Choose the symptoms to guess the disease. NOTE: the App is not perfect.")
st.write("The quality of the answer may only represent the dataset the model was tested on. "
"Though it realy more functions as a structure and could be updated variously for best practice or personal needs!")

st.subheader("Choose Symptoms")
chosen_symptoms = st.multiselect("Select symptoms", symptoms_list)




if st.button("Generate"):
    st.write("predictions:")
    for i, j in full_answer(chosen_symptoms).items():
        st.write(f"{i}: {j}")

#   takes more time as the function is actuall calculating all the patients for a disease and all the symptoms for a disease! 
if st.button("Full Data"):
    tuple = get_all_data(next(iter(full_answer(chosen_symptoms))))
    st.write(tuple[0])
    for i, j in tuple[1].items():

        st.write(f"{i}: {j}%")




#   TODO: further ideas:
#   *** standartize the way to update diseases / symptoms in the database!! ***
#   1. use LLM to explain main disease.
#   2. use LLM to group symptoms, if necessary (according to relevance OR if they appear to be describing more diseases) AND perform two searches

