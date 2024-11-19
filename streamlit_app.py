import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

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
    return model.predict(df)

def main():
    st.title('Jaya Jaya Maju Employee Attrition Prediction')

    with st.container():
        col_gender, col_age, col_marital = st.columns(3)
        with col_gender:
            gender = st.radio('Gender', options=['Male', 'Female'])
        with col_age:
            age = st.number_input('Age', min_value=18, max_value=60)
        with col_marital:
            marital_status = st.selectbox('Marital Status',
                ('Single', 'Married', 'Divorced'))
    
    with st.container():
        col_education, col_edu_field = st.columns(2)
        with col_education:
            education = st.selectbox('Education',
                ('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))
        with col_edu_field:
            education_field = st.selectbox('Education Field',
                ('Human Resources', 'Life Sciences', 'Marketing', 'Medical',
                 'Technical Degree', 'Other'))

    st.text('')
    st.text('')
    
    with st.container():
        col_distance, col_business_travel = st.columns(2)
        with col_distance:
            distance_from_home = st.number_input('Distance From Home to Work (in Km)', step=1)
        with col_business_travel:
            business_travel = st.selectbox('Business Travel',
                ('Non-Travel', 'Travel Rarely', 'Travel Frequently'))

    with st.container():
        col_dept, col_job_role, col_job_level = st.columns([2, 2, 1])
        with col_dept:
            department = st.selectbox('Department',
                ('Human Resources', 'Research & Development', 'Sales'))
        with col_job_role:
            job_role = st.selectbox('JobRole',
                ('Human Resources', 'Sales Executive', 'Sales Representative',
                 'Healthcare Representative', 'Research Scientist',
                 'Laboratory Technician', 'Manager', 'Manufacturing Director',
                 'Research Director'))
        with col_job_level:
            job_level = st.selectbox('Job Level',
                ('1', '2', '3', '4', '5'))

    st.text('')
    st.text('')
    
    with st.container():
        col_hourly_rate, col_daily_rate, col_monthly_rate = st.columns(3)
        with col_hourly_rate:
            hourly_rate = st.number_input('Hourly Rate', step=1)
        with col_daily_rate:
            daily_rate = st.number_input('Daily Rate', step=100)
        with col_monthly_rate:
            monthly_rate = st.number_input('Monthly Rate', step=1000)

    with st.container():
        col_monthly_income, col_percent_salary_hike = st.columns(2)
        with col_monthly_income:
            monthly_income = st.number_input('Monthly Income', step=100)
        with col_percent_salary_hike:
            percent_salary_hike = st.number_input('Percent Salary Hike (%)', step=1)
    
    with st.container():
        col_standard_hours, col_over_time = st.columns(2)
        with col_standard_hours:
            standard_hours = st.number_input('Standard Hours', value=80)
        with col_over_time:
            over_time = 'Yes' if st.checkbox('Over Time') else 'No'
    
    st.text('')
    st.text('')
    
    with st.container():
        col_job_sat, col_env_sat, col_rel_sat = st.columns(3)
        with col_job_sat:
            job_satisfaction = st.select_slider('Job Satisfaction',
                options=['Low', 'Medium', 'High', 'Very High'])
        with col_env_sat:
            environment_satisfaction = st.select_slider('Environment Satisfaction',
                options=['Low', 'Medium', 'High', 'Very High'])
        with col_rel_sat:
            relationship_satisfaction = st.select_slider('Relationship Satisfaction',
                options=['Low', 'Medium', 'High', 'Very High'])
    
    with st.container():
        col_job_inv, col_performance_rate, col_work_life_balance = st.columns(3)
        with col_job_inv:
            job_involvement = st.select_slider('Job Involvement',
                options=['Low', 'Medium', 'High', 'Very High'])
        with col_performance_rate:
            performance_rating = st.select_slider('Performance Rating',
                options=['Low', 'Good', 'Excellent', 'Outstanding'])
        with col_work_life_balance:
            work_life_balance = st.select_slider('Work Life Balance',
                options=['Low', 'Good', 'Excellent', 'Outstanding'])

    st.text('')
    st.text('')
    
    with st.container():
        col_stock_opt_lvl, col_num_companies_worked, col_training = st.columns(3)
        with col_stock_opt_lvl:
            stock_option_level = st.selectbox('Stock Option Level', ('0', '1', '2', '3'))
        with col_num_companies_worked:
            num_companies_worked = st.number_input('Number of Companies Worked', step=1)
        with col_training:
            training_times_last_year = st.number_input('Training Times Last Year', step=1)
    
    with st.container():
        col_tot_work_years, col_years_at_company, col_years_curr_role = st.columns(3)
        with col_tot_work_years:
            total_working_years = st.number_input('TotalWorkingYears', step=1)
        with col_years_at_company:
            years_at_company = st.number_input('Years at Company', step=1)
        with col_years_curr_role:
            years_in_current_role = st.number_input('Years in Current Role', step=1)
    
    with st.container():
        col_years_last_promotion, col_years_curr_manager = st.columns(2)
        with col_years_last_promotion:
            years_since_last_promotion = st.number_input('Years Since Last Promotion', step=1)
        with col_years_curr_manager:
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

    @st.dialog('Prediction Result')
    def prediction(output):
        if output == 1:
            st.subheader('Status Attrition: Yes', divider='red')
        else:
            st.subheader('Status Attrition: No', divider='green')
    
    if st.button('✨ Predict'):
        data_input = data_preprocessing(df)
        output = model_predict(data_input)
        prediction(output)

    year_now = datetime.date.today().year
    year = year_now if year_now == 2024 else '2024 - ' + str(year_now)
    name = "[Andrew Benedictus Jamesie](http://linkedin.com/in/andrewbjamesie 'Andrew Benedictus Jamesie | LinkedIn')"
    copyright = 'Copyright © ' + str(year) + ' ' + name
    st.caption(copyright)

if __name__ == '__main__':
    main()
