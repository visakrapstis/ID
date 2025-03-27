### Disease Prediction Model Project
#### State of Work
The Disease Prediction Model project is currently in progress, with the following steps completed or in development:

1. **Planning**: The project plan has been outlined, including the scope, objectives, and timeline.
2. **Scraping Database from the Internet**: A comprehensive database of disease-related information has been scraped from internet-site (https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html).
3. **Data Analyzing and Cleaning**: The scraped data has been analyzed and cleaned to ensure accuracy and consistency.
4. **Introducing LLM to Optimize the Dataset**: A Large Language Model has been introduced to optimize the dataset by incorporating frequencies of symptoms for given diseases, enhancing the dataset's quality and relevance. (prompt engineering, to combine the simulated data with actual data NLP (cosine similarity))
5. **Preparing ML (Dummies) Dataset**: A Machine Learning (ML) dataset has been prepared, utilizing the optimized data to create a solid foundation for model development.
6. **Augmentation of the new  Dataset**: The new dataset has been augmented to simulate over 600k unique patients, taking into account the number of disease occurrences and AI-assumed frequencies of symptoms for each disease.
7. **ML-Log-Reg Model**: A Logistic Regression model has been developed and trained using the dataset.
8. **Deep Learning Model with Tensorflow**: A Deep Learning model has been created using Tensorflow, providing a more complex and accurate learning capability.
9. **Model Testing**: The developed models were tested and evaluated to discover their accuracy and reliability.
10. **Streamlit App**: A Streamlit app was developed to deploy the DL-model, providing a user-friendly interface for disease prediction and visualization.

#### Pictures

<img src="pics/symptom_checker_2.jpg" style="width:400px;height:400px;">   <img src="pics/symptom_checker_3.jpg" style="width:400px;height:450px;">


#### Next Steps
The project can be continued with the following steps:

* Standardizing the database (disease/symptoms) updating
* Refining the models based on testing results
* train NN-Model (with more epochs) on a better quality diseases datset
* 
* Deploying the Streamlit app for public access
* Continuously updating and improving the models with new data and feedback
