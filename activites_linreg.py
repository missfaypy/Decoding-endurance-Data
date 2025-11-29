"""

In this script we train a linear regression model on running data. To be more precise we ask the question,
how precisely the duration ("moving time") of a run can be predicted by the distance of the run.
Additionally, the model is plotted on testing data and mae, mse etc. are calculated to determine
the models quality.

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as pplt


#load the .csv; only the columns "Activity Type", "Elapsed Time", "Moving Time" and "Distance".
def load_df():
    try:
        columns = ["Activity Type", "Elapsed Time", "Moving Time", "Distance"]
        df = pd.read_csv("/Users/jonassenn/Documents/vscode/Main/FHGR/datascience_project2/activities_dataset.csv", usecols=columns)
        return df
    except FileNotFoundError:
        print("File 'activities_dataset.csv' couldn't be found.")
        return None



def main():
    #create dataframe
    df = load_df()

    #change the distance datapoint which are written with an "," into datapoints with an "."
    #(all "swims" were registered with a comma in meters (e.g. 1,600m for 1600m), so changing the comma into a period will turn meters in to km(e.g. 1,600m -> 1.600km)
    df["Distance"] = df["Distance"].str.replace(",", ".")

    #turn the distance datapoints which are strings into floats
    convert_dict = {"Distance": float}
    df = df.astype(convert_dict)

    #Turn data in "Moving Time" from seconds to minutes
    df["Moving Time"] = df["Moving Time"] / 60

    #check for missing values
    print(df.isnull().sum())

    #create dataframe just with runs
    runs_df = df[df["Activity Type"] == "Run"]



    ##### TRAINING MODEL #####
    X = runs_df[["Distance"]]
    y = runs_df["Moving Time"]

    #splitting dfs into training/testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n")
    print(f'coefficient = {model.coef_} \nintercept = {model.intercept_}')

    #predict "moving time" for the X_test data
    y_pred = model.predict(X_test)

    #compare y_pred to y_test
    actual_compared_to_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print("\n")
    print(actual_compared_to_prediction)

    #calculate mae, mse, rmse and r2
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2 Score: ", r2)


    ### VISUALIZATION ###
    #plot the linear regression line
    pplt.plot(X_test, y_pred, label="linear regression", color="red")
    #plot the testing data points
    pplt.scatter(X_test, y_test, label="(testing) data points")

    pplt.xlabel("Distance (km)")
    pplt.ylabel("Moving Time (minutes)")
    pplt.legend()
    pplt.show()





if __name__ == "__main__":
    main()
