import random
import time
import numpy as np #better for arrays


#Note: I left some debug code to show my progress. 
#References: I put all the general references in my report since I used them throughout the code and not at any particular part. 
# I also inputted some specific references in the code below. 


#For this function, I used this to learn how to use argmin:https://www.geeksforgeeks.org/numpy-argmin-python/ 
#For KNN, I have inlcuded many links I referenced and  learned from in the report. 

def leave_one_out_cross_validation(data, current_set, feature_to_add): #nearest neighbor, cs170 demo function in video
    #first column -> class     
    size = data.shape
    dsizecol = size[0] #num of rows/data points

    dcopy = np.copy(data) #copy of data so original data doesn't get modified 

    #set features to 0 if they arent in the current set and if its not the current feature being tested
    for i in range(1, dcopy.shape[1]):
        if i not in current_set and i != feature_to_add:
            dcopy[:,i]= 0

    number_correctly_classified = 0
    
    for i in range(dsizecol): #loop through each row
        object_to_classify = dcopy[i, 1:] #extract feature values of current point
        label_object_to_classify = dcopy[i, 0] #extract label of current point

        allDistances = np.sum((dcopy[:, 1:] - object_to_classify) ** 2, axis=1) #computes euclidean distances between this point and all other points
        allDistances = np.sqrt(allDistances) #takes square root (part of previous)

        allDistances[i] = np.inf #ignore self so it doesnt get chosen as closest neighbor

        #finds nearest neighbor
        nearest_neighbor_location = np.argmin(allDistances) #index of smallest value
        nearest_neighbor_label = dcopy[nearest_neighbor_location, 0] #label of the neartest neighbor

        if label_object_to_classify == nearest_neighbor_label: #count to check if classifcation was accurate/correct
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified/(dsizecol) #classification calculation

        #print(f"Looping over i, at the {i} location")
        #print(f"The {i}th object is in class {label_object_to_classify}")



    return accuracy

    



def forwardSelection(data): #data is the data set youre intaking
    startTime = time.time() #start timer

    size = data.shape #get num of rows and column in dataset
    dsize = size[1] #get number of columns (where attributes are)

    current_set_of_features = [] #initialize empty list so we can keep track of items
    global_accuracy = 0 #best overall accuracy found
    best_set = [] #best optimal sub set

    #initial accuracy with no features
    initialAcc = leave_one_out_cross_validation(data, current_set_of_features, None)
    print(f"Running nearest neighbor with no features, using 'leave-one-out' evaluation. I get an accuracy of {initialAcc*100:.1f} %")
    print("Beginning search.\n")

    for i in range(1, dsize): #first column is just label so it doesn't count, go up to last column
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        #iterate over all possible features to add
        for k in range(1, dsize): #these nested loops helps us traverse through the search space  
            if k not in current_set_of_features:  #dont add duplicate features, make sure each feature is added only once
                accuracy = leave_one_out_cross_validation(data,current_set_of_features, k) #we are looking to remember highest num
                print(f"Using feature(s) {(current_set_of_features + [k])} accuracy is {accuracy*100:.1f} %")
               
                #get max (local accuracy) -> getting the best accuracy in k
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #k feature gave us this accuracy so we want to add it
    
        


        
        #if new feature improves accuracy, add it to current set
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level) #add new feature to current set since we chose it
            if global_accuracy < best_so_far_accuracy:
                global_accuracy = best_so_far_accuracy
                best_set = current_set_of_features.copy()
                print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy*100:.1f}%")
        else:
            print(f"Warning, Accuracy has decreased! Continuing search in case of local maxima") #ask if this is right

    endTime = time.time() #end timer
    print(f"Finished Search! The best feature subset is {best_set}, which has an accuracy of {global_accuracy*100:.1f} %. The total time taken to execute is {endTime - startTime:.3f} seconds.")
 
def backwardElimination(data):

    startTime = time.time() #start timer

    size = data.shape
    dsize = size[1]

    #start with list of all features
    current_set_of_features =[]
    selected_set =[]
    for ft in range(1, dsize):
        current_set_of_features.append(ft)
    bestSet = current_set_of_features.copy()

    global_accuracy = leave_one_out_cross_validation(data, current_set_of_features, None)
    print(f"\nRunning nearest neighbor with all {dsize - 1} features, using 'leave-one-out' evaluation, I get an accuracy of {global_accuracy * 100:.1f}%")
    print("Beginning search.\n")

    for i in range(1,dsize):
        print(f"On the {i}th level of the search tree")
        feature_to_delete_at_this_level = None
        best_so_far_accuracy = 0 #float('inf') #global accuracy

        #try removing each feature and check accuracy 
        for k in range(1, dsize):
            if k in current_set_of_features: #make sure the feature exists in the set before you can delete it
                temp_set = current_set_of_features.copy()
                temp_set.remove(k)

                accuracy = leave_one_out_cross_validation(data,temp_set, None) #calculate accuracy
                print(f"Using feature(s) {temp_set} accuracy is {accuracy*100:.1f} %")
                
                #get best accuracy and which feature causes it
                if accuracy >= best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_delete_at_this_level = k
        
        # check if feature was selected
        if feature_to_delete_at_this_level is not None:
            current_set_of_features.remove(feature_to_delete_at_this_level) #remove feature with the least accuracy
            #if removing feature improved accuracy, update global accuracy
            if best_so_far_accuracy > global_accuracy: 
                global_accuracy = best_so_far_accuracy
                bestSet = current_set_of_features.copy() #save current set since we have better accuracy now
                print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy * 100:.1f}%")
            else:
                print(f"Warning, Accuracy has decreased! Continuing search in case of local maxima") #if accruacy didnt improve

            

    
    endTime = time.time() #end timer
    print(f"Finished search! The best feature subset is {bestSet}, which has an accuracy of {global_accuracy * 100:.1f}%. Time taken to execute is {endTime - startTime:.3f} seconds.")




if __name__ == "__main__":
    print("Welcome to Shivani Nandakumar's Feature Selection Algorithm!")
    dataSet = input("Type in the name of the file to test: ")
    data = np.loadtxt(dataSet) #load dataset

    instances = data.shape[0]
    features = data.shape[1] - 1 #not including label column

    print(f"This dataset has {features} features (not including the class atttribute), with {instances} instances.")
    print("Type the number of the algorithm you want to run.")
    print(" (1) Forward Selection")
    print(" (2) Backward Elimination")
    option = input("Option: ")

    if option == "1":
        forwardSelection(data)
    elif option == "2":
        backwardElimination(data)

    