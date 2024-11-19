import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def data_preprocessing(data_input):
    df = pd.read_csv('employee_data_cleaned.csv')
    df = df.drop(columns=['EmployeeId', 'Attrition'], axis=1)
    df = pd.concat([df, data_input])
    
    numerical = df.select_dtypes(exclude='object').columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    
    df[categorical] = df[categorical].apply(LabelEncoder().fit_transform)
    df[numerical] = MinMaxScaler().fit_transform(df[numerical])
    
    return df.tail(1).to_numpy()

def model_predict(df):
    model = joblib.load('model_gb.joblib')
    pred = model.predict(df)

    return 'Attrition: Yes' if pred == 1 else 'Attrition: No'

def main():
    st.title('Jaya Jaya Maju Employee Attrition Prediction')

    with st.container():
        col_age, col_gender, col_marital = st.columns(3)
        with col_age:
            age = st.number_input('Age', min_value=18, max_value=60)
        with col_gender:
            gender = st.radio('Gender', options=['Male', 'Female'])
        with col_marital:
            marital_status = st.selectbox('Marital Status',
                ('Single', 'Married', 'Divorced'))
    
    education = st.selectbox('Education',
        ('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))
    education_field = st.selectbox('Education Field',
        ('Human Resources', 'Life Sciences', 'Marketing', 'Medical',
         'Technical Degree', 'Other'))

    distance_from_home = st.number_input('Distance From Home (in Km)', step=1)
    job_role = st.selectbox('JobRole',
        ('Human Resources', 'Sales Executive', 'Sales Representative',
         'Healthcare Representative', 'Research Scientist',
         'Laboratory Technician', 'Manager', 'Manufacturing Director',
         'Research Director'))
    department = st.selectbox('Department',
        ('Human Resources', 'Research & Development', 'Sales'))
    job_level = st.selectbox('Job Level',
        ('1', '2', '3', '4', '5'))
    business_travel = st.selectbox('Business Travel',
        ('Non-Travel', 'Travel Rarely', 'Travel Frequently'))

    daily_rate = st.number_input('Daily Rate', step=100)
    hourly_rate = st.number_input('Hourly Rate', step=1)
    monthly_income = st.number_input('Monthly Income', step=100)
    monthly_rate = st.number_input('Monthly Rate', step=1000)
    percent_salary_hike = st.number_input('Percent Salary Hike (%)', step=1)

    standard_hours = st.number_input('Standard Hours', value=80)
    over_time = 'Yes' if st.checkbox('Over Time') else 'No'
    job_satisfaction = st.select_slider('Job Satisfaction',
        options=['Low', 'Medium', 'High', 'Very High'])
    environment_satisfaction = st.select_slider('Environment Satisfaction',
        options=['Low', 'Medium', 'High', 'Very High'])
    relationship_satisfaction = st.select_slider('Relationship Satisfaction',
        options=['Low', 'Medium', 'High', 'Very High'])
    job_involvement = st.select_slider('Job Involvement',
        options=['Low', 'Medium', 'High', 'Very High'])
    performance_rating = st.select_slider('Performance Rating',
        options=['Low', 'Good', 'Excellent', 'Outstanding'])
    work_life_balance = st.select_slider('Work Life Balance',
        options=['Low', 'Good', 'Excellent', 'Outstanding'])

    stock_option_level = st.selectbox('Stock Option Level', ('0', '1', '2', '3'))
    num_companies_worked = st.number_input('Number of Companies Worked', step=1)
    training_times_last_year = st.number_input('Training Times Last Year', step=1)

    total_working_years = st.number_input('TotalWorkingYears', step=1)
    years_at_company = st.number_input('Years at Company', step=1)
    years_in_current_role = st.number_input('Years in Current Role', step=1)
    years_since_last_promotion = st.number_input('Years Since Last Promotion', step=1)
    years_with_curr_manager = st.number_input('Years with Current Manager', step=1)
    
    data = [[age, business_travel, daily_rate, department, distance_from_home,
        education, education_field, environment_satisfaction, gender,
        hourly_rate, job_involvement, int(job_level), job_role, job_satisfaction,
        marital_status, monthly_income, monthly_rate, num_companies_worked,
        over_time, percent_salary_hike, performance_rating,
        relationship_satisfaction, standard_hours, int(stock_option_level),
        total_working_years, training_times_last_year, work_life_balance,
        years_at_company, years_in_current_role, years_since_last_promotion,
        years_with_curr_manager]]

    df = pd.DataFrame(data, columns=[
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'])
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    if st.button('✨ Predict'):
        with st.container():
            data_input = data_preprocessing(df)
            output = model_predict(data_input)

            st.subheader('Status ' + output, divider='gray')

    year_now = datetime.date.today().year
    year = year_now if year_now == 2024 else '2024 - ' + str(year_now)
    name = "[Andrew Benedictus Jamesie](http://linkedin.com/in/andrewbjamesie 'Andrew Benedictus Jamesie | LinkedIn')"
    copyright = 'Copyright © 2024-' + str(year) + ' ' + name
    st.caption(copyright)

if __name__ == '__main__':
    main()
