import matplotlib.pyplot as plt
import numpy as np


def descriptive_stats(X_train, X_test, y_train, y_test):
    # Summary statistics for features in X_train
    summary_stats = X_train.describe()
    print(summary_stats)
    
    # Plot histograms for each feature in X_train
    for column in X_train.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(X_train[column], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
    # Unique classes and their counts in y_train
    unique_classes_train, counts_train = np.unique(np.argmax(y_train.values, axis=1), return_counts=True)
    class_counts_train = dict(zip(unique_classes_train, counts_train))
    
    print("Classes and their counts in y_train:")
    for class_index, count in class_counts_train.items():
        print(f"Class {class_index}: Count - {count}")
    
    # Unique classes and their counts in y_test
    unique_classes_test, counts_test = np.unique(np.argmax(y_test.values, axis=1), return_counts=True)
    class_counts_test = dict(zip(unique_classes_test, counts_test))
    
    print("\nClasses and their counts in y_test:")
    for class_index, count in class_counts_test.items():
        print(f"Class {class_index}: Count - {count}")
    
    return summary_stats