"""
This program uses a Linear Regression Model to predict the rating of the University into which a
student could get into, based upon certain scores specifications as attributes.
@author Anish Krishnan Ganesh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

# Read Admission_Predict.csv file using Pandas
data = pd.read_csv("Admission_Predict.csv")

# Declare and Assign variables to commonly used attributes and labels
x_value = "GPA"
predict = "University Rating"

# Create attributes array and labels array
X = np.array(data.drop([ "Serial No.", predict ], 1))
y = np.array(data[predict])

best_acc = 0
# Iterate through loop to train and store best possible model in pickle file
"""for _  in range(1000):

    # Split training data and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Create a linear model that uses the Linear Regression algorithm
    linear = linear_model.LinearRegression()
    # Train the model with the training data stored
    linear.fit( x_train, y_train )
    # Test the accuracy of the model and store it in variable acc
    acc = linear.score( x_test, y_test )
    #print("Accuracy of model:", acc)

    # Check if the current accuracy is greater than the best accuracy
    if acc > best_acc:
        best_acc = acc
        # Override file with the higher accuracy model
        with open("linear_model.pickle", "wb") as f:
            pickle.dump(linear, f)
            pickle.dump(acc, f)"""

# Load the best accuracy model from the linear_model.pickle file
with open("linear_model.pickle", "rb") as f:
    linear = pickle.load(f)
    best_acc = pickle.load(f)

# Display required information to the user
"""print("This program trains a Linear Regression Model, and predicts the rating of the University in which the\nstudent would get an admission, depending on certain attributes.")
print("Accuracy of the Model:", best_acc)

# Prompt user to enter attribute data for prediction
gre_score = int(input("\nEnter the GRE Score( out of 340 ): "))
toefl_score = int(input("Enter the TOEFL Score( out of 120 ): "))
sop = float(input("Enter the quality of the Statement of Purpose( out of 5 ): "))
lor = float(input("Enter the rating of the Letter of Recommendation( out of 5 ): "))
gpa = float(input("Enter the Grade Point Average( out of 10 ): "))
research = int(input("Enter whether the student has submitted the Research Paper( 0 or 1 )"))

# Make the prediction using the trained linear model and display the output
prediction = linear.predict( [[gre_score, toefl_score, sop, lor, gpa, research]] )
pred_val = int(round(prediction[0]))

if pred_val < 1 :
    print("\nThis student would not get an admission in any University with his/her current score specification.")
else :
    univ_type = { 1 : "Tier 5", 2 : "Tier 4", 3 : "Tier 3", 4 : "Tier 2", 5 : "Tier 1" }
    print("\nThis student would get an admission in a University of rating", str(pred_val) + ", a", univ_type.get(pred_val), "University.")"""

print("Coefficients:", linear.coef_)
print("Intercept:", linear.intercept_)

"""predictions = linear.predict( x_test )
for x in range(len(predictions)):
    print( round(predictions[x]), x_test[x], y_test[x] )

# Plot scatter plot with independent and dependant variable
style.use("ggplot")
plt.scatter(data[[x_value]], data[[predict]])
plt.xlabel(x_value)
plt.ylabel(predict)
plt.show()"""