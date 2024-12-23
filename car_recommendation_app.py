import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    return pd.read_csv("USA_cars_datasets.csv")  # Use the relative path

car_data = load_data()

# Title and description
st.title("🚗 Car Recommendation System")
st.markdown("Explore and predict car prices based on features like mileage, year, and more!")

# Sidebar Filters
st.sidebar.header("Filter Cars")
brands = st.sidebar.multiselect("Brand:", options=car_data["brand"].unique(), default=car_data["brand"].unique())
min_price, max_price = st.sidebar.slider("Price Range:", int(car_data["price"].min()), int(car_data["price"].max()), (5000, 30000))
max_mileage = st.sidebar.slider("Maximum Mileage:", 0, int(car_data["mileage"].max()), int(car_data["mileage"].mean()))
year_range = st.sidebar.slider("Year Range:", int(car_data["year"].min()), int(car_data["year"].max()), (2005, 2020))

# Filter data based on selections
filtered_data = car_data[(
    car_data["brand"].isin(brands)) & 
    (car_data["price"].between(min_price, max_price)) & 
    (car_data["mileage"] <= max_mileage) & 
    (car_data["year"].between(*year_range))
]

# Display filtered results
st.header("Filtered Cars")
if not filtered_data.empty:
    st.dataframe(filtered_data)
else:
    st.warning("No cars match your criteria. Please adjust the filters.")

# Visualizations
st.header("Data Visualizations")
if st.checkbox("Show Brand Distribution"):
    brand_counts = car_data["brand"].value_counts()
    st.bar_chart(brand_counts)

if st.checkbox("Show Price vs Mileage Scatter Plot"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="mileage", y="price", hue="brand", data=car_data, alpha=0.7)
    plt.title("Price vs Mileage")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    st.pyplot(plt)

# Regression Analysis: Train model
st.header("📈 Predict Car Price")

# Features for the model
X = car_data[['mileage', 'year', 'condition', 'brand', 'model', 'title_status', 'color', 'state', 'country']]  # Added new features
y = car_data['price']  # Target variable

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

# Display model performance
st.write(f"### Regression Model R² Score: {r2:.2f}")

# Scatter Plot of Predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.7, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
st.pyplot(plt)

# User input for prediction
mileage_input = st.number_input("Enter Mileage:", min_value=0, step=1000)
year_input = st.number_input("Enter Year:", min_value=1990, step=1)
condition_input = st.selectbox("Select Condition:", options=car_data["condition"].unique())
brand_input = st.selectbox("Select Brand:", options=car_data["brand"].unique())
model_input = st.selectbox("Select Model:", options=car_data["model"].unique())
title_status_input = st.selectbox("Select Title Status:", options=car_data["title_status"].unique())
color_input = st.selectbox("Select Color:", options=car_data["color"].unique())
state_input = st.selectbox("Select State:", options=car_data["state"].unique())
country_input = st.selectbox("Select Country:", options=car_data["country"].unique())

if st.button("Predict"):
    # Prepare the user input as a dataframe
    user_input = pd.DataFrame({
        "mileage": [mileage_input], 
        "year": [year_input], 
        "condition": [condition_input],
        "brand": [brand_input], 
        "model": [model_input], 
        "title_status": [title_status_input], 
        "color": [color_input], 
        "state": [state_input], 
        "country": [country_input]
    })
    
    # One-hot encode user input
    user_input = pd.get_dummies(user_input, drop_first=True)

    # Align the user input with the training data (handling missing columns)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Predict the price
    predicted_price = model.predict(user_input)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")

# Media Content
st.header("Media")
st.video("https://www.youtube.com/watch?v=SDvRK7v7q9I")  # Example car video
st.image("C:/Users/ishan/Downloads/streamlit-20241202T033733Z-001/streamlit/yuvraj-singh-tmAynVA_ihE-unsplash.jpg", caption="For the love of carzzz!")

# Footer
st.sidebar.info("Built with ❤️ using Streamlit and Machine Learning")
