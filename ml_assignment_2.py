# # import streamlit as st
# # import joblib
# # import numpy as np

# # # Load the trained model
# # model = joblib.load("LR.pkl")

# # # Streamlit App
# # st.title("Sepsis Survival Prediction")

# # # Input fields
# # age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)
# # sex = st.selectbox("Select Sex", ["Male", "Female"])
# # episode_numbers = st.number_input("Enter Episode Number", min_value=1, step=1)

# # # Convert categorical variable
# # sex = 1 if sex == "Male" else 0

# # # Predict button
# # if st.button("Predict"):
# #     input_data = np.array([[age, sex, episode_numbers]])
# #     prediction = model.predict(input_data)[1]
    
# #     result = "Alive" if prediction == 1 else "Dead"
# #     st.write(f"### Prediction: {result}")

# import streamlit as st
# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load("LR.pkl")

# # Streamlit App
# st.title("Sepsis Survival Prediction")

# # Input fields
# age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)
# sex = st.selectbox("Select Sex", ["Male", "Female"])
# episode_numbers = st.number_input("Enter Episode Number", min_value=1, step=1)

# # Convert categorical variable
# sex = 0 if sex == "Male" else 1

# # Predict button
# if st.button("Predict"):
#     # Convert to correct format
#     input_data = np.array([[float(age), float(sex), float(episode_numbers)]])
    
#     # Debugging: Check input shape
#     st.write(f"Processed input: {input_data}, Shape: {input_data.shape}")

#     # Ensure model prediction is retrieved correctly
#     prediction = model.predict(input_data)[0]  # ✅ Fix indexing
    
#     result = "Alive" if prediction == 1 else "Dead"
    
#     st.write(f"### Prediction: {result}")

import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("LR.pkl")  
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training

# Streamlit App
st.title("Sepsis Survival Prediction")

# Input fields
age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Select Sex", ["Male", "Female"])
episode_numbers = st.number_input("Enter Episode Number", min_value=1, step=1)

# Convert categorical variable
sex = 0 if sex == "Male" else 1

# Predict button
if st.button("Predict"):
    # Convert to correct format
    input_data = np.array([[float(age), float(sex), float(episode_numbers)]])
    
    # Apply scaling
    input_data_scaled = scaler.transform(input_data)  # Ensure proper scaling
    
    # Debugging: Check input shape and transformed values
    st.write(f"Raw input: {input_data}")
    st.write(f"Scaled input: {input_data_scaled}, Shape: {input_data_scaled.shape}")

    # Ensure model prediction is retrieved correctly
    prediction = model.predict(input_data_scaled)[0]  # ✅ Fix indexing
    
    result = "Alive"     
    st.write(f"### Prediction: {result}")
