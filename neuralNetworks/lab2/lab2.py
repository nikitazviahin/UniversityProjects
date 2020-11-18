import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train_data.csv", sep=',', header=0)
cleaned_data = np.array([train_data['Feature 1'].tolist(), train_data['Feature 2'].tolist()]).T

groups = train_data.groupby('Class') # DataFrameGroupBy type, groups all classes and 
number_of_classes = len(groups)
dictionary_of_sum = {}
sigma = 0.3

print(pd.DataFrame(groups.get_group('A')))
print(pd.DataFrame(groups.get_group('B')))

test_examples  = [
    [1, 0],
    [3, 4],
    [12, 10],   
]

for example in test_examples:    

    index_current_example = 0
    for k in groups.groups.keys():
        dictionary_of_sum[k] = 0
        
        number_of_data_point_from_class_k = len(groups.get_group(k))
        

        
        product = 0.0
        
        for _ in range(number_of_data_point_from_class_k):
            tempx = (example[0] - cleaned_data[index_current_example][0]) ** 2
            tempy = (example[1] - cleaned_data[index_current_example][1]) ** 2
            
            temp_sum = np.exp(-tempx / np.power(sigma, 2)) + np.exp(-tempy / np.power(sigma, 2))
            
            product += temp_sum
            index_current_example += 1
        
        dictionary_of_sum[k] = product / number_of_data_point_from_class_k
    
    classified_class = str(max(dictionary_of_sum, key=dictionary_of_sum.get))
    
    groups = train_data.groupby('Class')
    
    fig, ax = plt.subplots()
    
    for name, group in groups:
        ax.plot(group['Feature 1'], group['Feature 2'], marker='X', linestyle='', ms=15, label=name)
        
    # Draw the unclassified data point
    ax.plot(example[0], example[1], marker='X', linestyle='', ms=15)
    
    axes = plt.gca()

    plt.title('Відноситься до класу : ' + str(classified_class) )
    plt.xlabel('X')
    plt.ylabel('Y')

    # Giving a grid and show the plot
    plt.grid()
    plt.show()
