"""

In this script we train three classification models on sports data. The models learn to differentiate 
between three different "activity types" (running, swimming and bike riding) based on distance and moving time. 
Additionally, we visualize the data and calculate the f1_score of each model to determine its quality.

"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pplt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


#load the .csv; only certain columns.
def load_df():
    try:
        columns = ["Activity Type", "Moving Time", "Distance", "Max Speed"]
        df = pd.read_csv("/Users/jonassenn/Documents/vscode/Main/FHGR/datascience_project2/activities_dataset.csv", usecols=columns)
        return df
    except FileNotFoundError:
        print("File 'activities_dataset.csv' couldn't be found.")
        return None
    
def get_testing_activity():
    pass


def main():
    #create dataframe
    df = load_df()


    ##### QUALITY ASSESSMENT AND PREPROCESSING #####
    print(df.describe(), "\n")

    #change the distance datapoint which are written with an "," into datapoints with an "."
    #(all "swims" were registered with a comma in meters (e.g. 1,600m for 1600m), so changing the comma into a period will turn meters in to km(e.g. 1,600m -> 1.600km)
    df["Distance"] = df["Distance"].str.replace(",", ".")
    #turn the distance datapoints which are strings into floats
    convert_dict = {"Distance": float}
    df = df.astype(convert_dict)
    #Turn data in "Moving Time" from seconds to minutes
    df["Moving Time"] = df["Moving Time"] / 60

    # #get a list of all unique Activity Types
    # print("Activity types in data:\n", df["Activity Type"].unique())

    #create three new dataframes, only with the activity types Run, Ride and Swim
    run_df = df[df["Activity Type"].isin(["Run"])]
    swim_df = df[df["Activity Type"].isin(["Swim"])]
    ride_df = df[df["Activity Type"].isin(["Ride"])]

    #clean up some swimming distances; km -> m
    swim_df["Distance"] = swim_df["Distance"].apply(lambda x: x/1000 if x > 50 else x)

    #activity type "Ride" has some faulty distances, where distance = 0. We get rid of all those rows
    ride_df = ride_df[ride_df["Distance"] != 0]

    #create new dataframe from the three above
    new_df = pd.concat([run_df, swim_df, ride_df], ignore_index=True)

    #check for missing values
    print(new_df.isnull().sum())

    # transform activity type column to numbers
    labelencoder = LabelEncoder()
    new_df["Activity Type"] = labelencoder.fit_transform(new_df["Activity Type"])


    #####  DATA VISUALISATION #####
    # show amount of Run-, Ride- and Swim-Activities 
    sb.countplot(x = new_df["Activity Type"], color="seagreen")
    pplt.show()

    # Check correlation among other variables
    sb.heatmap(new_df.corr(), annot=True)
    pplt.show()

    #choose features we will use for model training
    features = ["Distance", "Moving Time"]

    y = new_df["Activity Type"]
    #get a df which has a column with a bool for each Activity type (0-2)
    y = pd.get_dummies(y)

    #create df we will use for training
    X = new_df[features]

    #visualize features
    sb.scatterplot(x=new_df["Distance"], y=new_df["Moving Time"], hue=new_df["Activity Type"])
    pplt.xlabel("Distance (km)")
    pplt.ylabel("Moving Time (minutes)")
    pplt.show()



    ##### MODEL TRAINING #####
    # Splitting data into training/testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


    # Create a RandomForestClassifier with n_estimators=500
    random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
    # Create a DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier(random_state=42)
    # Create a KNeighborsClassifier with n_neighbors=5
    k_neighbors = KNeighborsClassifier(n_neighbors=5)

    #create dict for easy looping over the three models
    models = {
                "Random Forest Classifier": random_forest,
                "Decision Tree Classifier": decision_tree,
                "K-Neighbors": k_neighbors
             }

    for name, model in models.items():
        model.fit(X_train.values, y_train.values)



    ##### MODEL EVALUATION #####      
    for name, model in models.items():
        pred = model.predict(X_test.values)
        sklearn_f1 = f1_score(y_test.values, pred, average='macro')
        print(f'F1_score of {name}: {sklearn_f1}')



    ##### TESTING  THE MODEL #####
    # activity = get_testing_activity()

    #create a testing activity
    activity = np.array([2, 30]).reshape(1, -1)

    #make predictions with the models
    for name, model in models.items():
        pred = model.predict(activity)
        activity_number = pd.DataFrame(pred).idxmax(axis=1)
        activity_type = labelencoder.inverse_transform(activity_number)[0]
        print(f'{name}: Diese Activity ist wahrscheinlich ein "{activity_type}".')   




if __name__ == "__main__":
    main()
