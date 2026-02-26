# Step 1: Import required libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Step 2: Create dataset (Hours studied vs Result)
# X = Hours studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# y = Result (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 1, 1, 1])

# Step 3: Create Logistic Regression model
model = LogisticRegression()

# Step 4: Train the model
model.fit(X, y)

# Step 5: Print model parameters
print("Intercept (b0):", model.intercept_[0])
print("Coefficient (b1):", model.coef_[0][0])

# Step 6: Predict for new value (3.5 hours)
new_hours = np.array([[3.5]])

prediction = model.predict(new_hours)
probability = model.predict_proba(new_hours)

print("Prediction (0=Fail, 1=Pass):", prediction[0])
print("Probability [Fail, Pass]:", probability[0])

# Step 7: Plot the sigmoid curve
X_test = np.linspace(0, 7, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

plt.scatter(X, y)
plt.plot(X_test, y_prob)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Sigmoid Curve")

plt.show()