from sklearn.model_selection import train_test_split

def preprocess_xiiot(data):
    while True:
        # Prompt user to choose the classification scenario
        print("Choose the classification scenario:")
        print("1. 2 classes\n2. 10 classes\n3. 19 classes")
        
        # Take user input for the classification scenario
        scenario_option = input("Enter the number for the classification scenario: ")

        # Check if the input is valid (1, 2, or 3)
        if scenario_option in ['1', '2', '3']:
            break  # Break the loop if the input is valid
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    # Logic to select the label based on the chosen scenario
    if scenario_option == '1':
        my_label = data.iloc[:, 61]  # Choose label_2 for 2 classes
    elif scenario_option == '2':
        my_label = data.iloc[:, 60]  # Choose label_10 for 10 classes
    else:
        my_label = data.iloc[:, 59]  # Choose label_19 for 19 classes

    features = data.iloc[:, :59]
    my_label = pd.get_dummies(my_label)

    # split data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(features, my_label, test_size=0.3, random_state=142)

    return X_train, X_test, y_train, y_test