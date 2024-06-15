# import streamlit packages

import streamlit as st 


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


import joblib





#calculate BMR (basal metabolic rate)
def bmr_calculate(gender, weight, height, age):
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return bmr


def daily_caloric(bmr,activity_level):
    activity_fac = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Super active": 1.9
    }
    
    return bmr * activity_fac[activity_level]



#streamlit part     
st.title("Track, Predict, and Understand the Health Implications of Your Weight Journey")
st.markdown("🏋️  💪  🏋️‍♂️")
st.title("Weight lose and Weight gain :- ")
st.markdown("---")
gender = st.selectbox("gender",[" male","female"])
age = st.slider(label="select your age",min_value=10, max_value=80,value=25,step=1)
height = st.slider(label="Height (cm)",min_value=120,max_value=200,value = 165,step=1)
weight = st.slider(label="Select your Weight in (KG)",min_value=30,max_value=300,value=65,step=1 )
activity_level = st.selectbox("Acticity Level",["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
goal_weight = st.slider(label="select weight goal",min_value=10,max_value=200,value=65,step=1)

goal_type = "gain" if goal_weight > weight else "lose"

# set a button
if st.button("Calculate"):
    bmr = bmr_calculate(gender, weight, height, age)
    daily_caloric_need =  daily_caloric(bmr,activity_level)  
    
    
    st.write(f" You Basal Metbolic Rate (BMR) is : {bmr:2f} caloric/day  ")
    st.write(f"You daily caloric need based on activity:{daily_caloric_need:2f} caloric/day")
    
    calorie_deficit_surplus=500
    
    
    if goal_type == "lose":
            daily_caloric_intake = daily_caloric_need - calorie_deficit_surplus
            weeks_needed = ((weight - goal_weight) * 7700) / (calorie_deficit_surplus * 7)
    else:
            daily_caloric_intake = daily_caloric_need + calorie_deficit_surplus
            weeks_needed = ((goal_weight - weight) * 7700) / (calorie_deficit_surplus * 7)
            
            
    st.write(f"To {goal_type} weight, you need to consume approximately {daily_caloric_intake:.2f} calories/day.")
    st.write(f"Estimated time to reach your goal weight: {weeks_needed:.2f} weeks.")
    
    
    
#dataset
data = {
    'age': [25, 30, 22, 35, 40],
    'exercise_frequency': [3, 1, 2, 4, 1],
    'hours_of_sleep': [7, 5, 6, 8, 4],
    'stress_level': [3, 8, 5, 2, 9],
    'previous_mental_health_issues': [0, 1, 0, 0, 1],
    'mental_health_score': [75, 45, 60, 80, 40]  # Target variable
}

df = pd.DataFrame(data)


# Define features and target variable
X = df.drop('mental_health_score', axis=1)
y = df['mental_health_score']


#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)





# Save the model
joblib.dump(model, 'mental_health_model.pkl')
 

 
 
     
                               
     
     
st.title("Check Mental Stress")     

# User inputs
age = st.slider(label="Age", min_value=0, max_value=100, value=25,step=1)
exercise_frequency = st.slider("Exercise Frequency (days per week)", min_value=0, max_value=7, value=3,step=1)
hours_of_sleep = st.slider("Hours of Sleep per Night", min_value=0, max_value=24, value=7,step=1)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
previous_mental_health_issues = st.selectbox("Previous Mental Health Issues", ["No", "Yes"])
  
  # Predict button
if st.button("Predict"):
    
      # Prepare input data
     input_data = pd.DataFrame([[age,  exercise_frequency, hours_of_sleep, stress_level, previous_mental_health_issues]], 
             columns=['age',  'exercise_frequency', 'hours_of_sleep', 'stress_level', 'previous_mental_health_issues'])
     input_data['previous_mental_health_issues'] = input_data['previous_mental_health_issues'].apply(lambda x: 1 if x == 'Yes' else 0)
     
     
     prediction = model.predict(input_data)
     
    
    # Display the result
     st.write(f"Predicted Mental Health Score: {prediction[0]}")    
  
        







    

  