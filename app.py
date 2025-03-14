import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.patches import ConnectionPatch


# App Title
st.set_page_config(page_title="AI-Powered Employee Performance Analyzer", layout="wide")
st.title("📊 AI-Powered Employee Performance Analyzer")

# App Description
st.write("This web application is an AI-powered employee performance analyzer that helps HR professionals to predict employee performance based on various features.")
st.info("""
### How to Use This Tool:
1. **Upload a CSV Dataset** with the following columns:
    - **Name**: Employee's full name.
    - **Age**: Age of the employee.
    - **Gender**: Male/Female/Other.
    - **Projects Completed**: Number of projects successfully completed.
    - Productivity (%): Employee's productivity percentage.
    - **Satisfaction Rate (%)**: Employee's satisfaction level in percentage.
    - **Feedback Score**: Overall feedback score from peers/supervisors.
    - **Department**: Department where the employee works.
    - **Position**: Job position/title of the employee.
    - **Joining Date**: Date when the employee joined the company.
    - **Salary**: Employee's total salary.

2. **Once uploaded, the dashboard will display key insights and visualizations.**
3. **Use the sidebar** to filter results by department and employee.
4. **Analyze performance trends** using dynamic charts and AI predictions.
5. **Predict employee performance** based on selected features using machine learning models.
6. **If necessary alter the dataset column by this in below github link for better peocessing** 
7. **GIT Repository**: [Employee Performance Analyzer](https://github.com/ranjithsurineni/AI-Performance-Analyzer.git)
8. **Author**: [Ranjith Kumar Surineni](https://www.linkedin.com/in/ranjith-kumar-surineni-b73b981b6/)
""")





# File uploader for dataset
st.sidebar.header("📂 Upload Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Initialize session state for dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = None

#----------------------------- Load Dataset -----------------------------#

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.dataset = df
    st.sidebar.success("✅ Dataset uploaded successfully!")

#----------------------------- Dataset Preview -----------------------------#

if st.session_state.dataset is not None:
    df = st.session_state.dataset
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    # Detect columns dynamically
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.error("⚠️ The dataset must have numerical features for analysis.")
    else:
        #----------------------------- Dynamic Feature Selection -----------------------------#

        # Select Categorical Columns
        st.sidebar.subheader("🔍 Select Categorical Columns* :")
        if categorical_cols:
            department_col = st.sidebar.selectbox("🏢 Select Department Column :", categorical_cols)
            employee_col = st.sidebar.selectbox("👤 Select Employee Column* :", categorical_cols)

            # Filter departments dynamically
            departments = df[department_col].unique().tolist()
            selected_department = st.sidebar.selectbox("🏢 Select Department :", departments)

            # Filter employees dynamically
            employees = df[df[department_col] == selected_department][employee_col].unique().tolist()
            selected_employee = st.sidebar.selectbox("👤 Select Employee :", employees)

        # Select Numerical Features
        st.sidebar.subheader("📌 Select Numerical Features for Train & Predict :")
        selected_feature = st.sidebar.selectbox("📈 Select Feature for Prediction :", numerical_cols)
        target_col = st.sidebar.selectbox("🎯 Select Target Column :", numerical_cols)

        # Filter employee and department data
        emp_data = df[df[employee_col] == selected_employee]
        dept_data = df[df[department_col] == selected_department]

        #----------------------------- Employee Performance Analysis -----------------------------#
        #--------------------------------pie chart starts-----------------------------#
        # Function to generate overall performance pie chart
        def plot_performance_pie(employee_name, selected_features):
            employee = df[df[employee_col] == employee_name].iloc[0]
            labels = [col for col in selected_features if col.lower() != 'salary' and col.lower() != 'age']
            values = [employee[col] for col in labels]
            
            # Filter out NaN values
            labels, values = zip(*[(label, value) for label, value in zip(labels, values) if not pd.isna(value)])
    

            # Convert values to percentages
            values_percentage = [v for v in values]

            # Generate a color palette dynamically
            colors = plt.cm.get_cmap('tab20c', len(labels)).colors

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                values_percentage, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90,
                textprops={'fontsize': 12}  # Improve text readability
            )
            ax.set_title(f"Overall Performance: {employee_name}")

            # Add legend & formatting
            plt.legend(wedges, labels, title="Metrics", loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))

            # Beautify text representation
            for text, autotext in zip(texts, autotexts):
                text.set_fontsize(12)
                text.set_color('black')
                text.set_fontweight('bold')
                autotext.set_fontsize(12)
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            return fig

        # Select Numerical Features
        st.sidebar.subheader("📌 Select Numerical Features for Pie chart")
        selected_features = st.sidebar.multiselect("📈 Select Features for Pie Chart", numerical_cols)

        # Filter employee and department data
        emp_data = df[df[employee_col] == selected_employee]
        dept_data = df[df[department_col] == selected_department]

        # Display the pie chart in Streamlit
        st.subheader(f"🍰 Overall Performance: {selected_employee}")

        if selected_features:
            pie_chart = plot_performance_pie(selected_employee, selected_features)
            st.pyplot(pie_chart)
        else:
            st.write("⚠ ⚡ Please select features to display the pie chart.✨")

        # Display employee salary
        employee_salary = df[df[employee_col] == selected_employee]['Salary'].values[0]
        monthly_salary = employee_salary / 12
        st.write(f"💵 **Monthly Salary:** ${monthly_salary:,.2f}")
        st.write(f"💵 **Annual Salary:** ${employee_salary:,.2f}")

        #--------------------------------piechart ends -----------------------------#
        
        #--------------------------------bar chart starts -----------------------------#
       
        # Improved Bar Chart: Employee vs Department Performance
        plt.figure(figsize=(12, 7))
        avg_dept_performance = dept_data[numerical_cols].mean()
        emp_performance = emp_data[numerical_cols].mean()
        performance_df = pd.DataFrame({'Metric': numerical_cols, 
                                    'Department Average': avg_dept_performance, 
                                    'Employee': emp_performance})

        # Reshaping for Seaborn
        performance_df = performance_df.melt(id_vars="Metric", var_name="Category", value_name="Value")

        # Using Seaborn horizontal barplot for better readability
        sns.set_style("whitegrid")
        ax = sns.barplot(data=performance_df, y="Metric", x="Value", hue="Category", palette="coolwarm")

        # Enhancing title and labels
        plt.title(f"📊 {selected_employee} vs Department Average Performance", fontsize=14, fontweight="bold")
        plt.xlabel("Performance Metrics", fontsize=12)
        plt.ylabel("Metrics", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Category", fontsize=10)

        # Adding data labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}', 
                        (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                        fontsize=9, color="black", va='center')

        # Display in Streamlit
        st.pyplot(plt)

        #--------------------------------bar chart ends -----------------------------#


        #----------------------------- Model Training Section starts -----------------------------#

        # Encode categorical feature dynamically
        le = LabelEncoder()
        df[department_col] = le.fit_transform(df[department_col])

        # Prepare data dynamically
        X = df[[selected_feature, department_col]]
        y = df[target_col]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select Model
        st.sidebar.subheader("🧠 Select Machine Learning Model")
        model_options = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100)
        }
        selected_model_name = st.sidebar.selectbox("Choose Model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        # Train Model
        selected_model.fit(X_train, y_train)

        # Save Model
        with open("./models/performance_model.pkl", "wb") as f:
            pickle.dump(selected_model, f)

        st.sidebar.success(f"✅ {selected_model_name} Trained Successfully!")

        # Model Evaluation
        y_pred = selected_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader(f"📊 Model Performance: {selected_model_name}")
        st.write(f"🔹 Mean Squared Error: {mse:.2f}")
        st.write(f"🔹 R-Squared Score: {r2:.2f}")

        #----------------------------- Model Training Section ends -----------------------------#

        #----------------------------- Prediction Section starts -----------------------------#

        st.subheader("🔮 Predict Employee Performance")
        user_input = st.number_input(f"Enter value for {selected_feature}", min_value=float(df[selected_feature].min()), max_value=float(df[selected_feature].max()), value=float(df[selected_feature].mean()))
        
        if st.button("Predict"):
            input_data = pd.DataFrame({selected_feature: [user_input], department_col: [df[df[employee_col] == selected_employee][department_col].values[0]]})
            prediction = selected_model.predict(input_data)
            predicted_value = round(prediction[0], 2)
            st.success(f"📌 Predicted {target_col}: {predicted_value}")
            
            # Employee Safety Status
            threshold = df[target_col].mean()  # Setting threshold as department average
            if predicted_value >= threshold:
                st.success("✅ Employee is SAFE based on performance prediction.")
            else:
                st.error("⚠️ Employee is at RISK based on performance prediction.")

        #----------------------------- Prediction Section ends -----------------------------#
