import numpy as np
from termcolor import cprint, colored
from SubFunctions.Model import Network
from SubFunctions.Evaluate import Evaluation_Metrics, Evaluation_Metrics1
import os



def train_test_split(data, train_size):

    labels = data['Labels']
    # Get the unique classes in the target variable 'y'
    num_classes = np.unique(labels)

    # Initialize empty lists to store training and testing data
    x_train1 = []
    x_train2 = []
    x_train3 = []
    x_train4 = []
    x_train5 = []
    x_train6 = []
    x_train7 = []
    x_train8 = []
    y_train = []

    x_test1 = []
    x_test2 = []
    x_test3 = []
    x_test4 = []
    x_test5 = []
    x_test6 = []
    x_test7 = []
    x_test8 = []
    y_test = []

    # Loop through each unique class
    for i in range(len(num_classes)):
        # Find indices of samples belonging to the current class
        indices = np.where(labels == num_classes[i])

        # Split the indices based on the specified 'train_size'
        train_index = indices[0][:int(len(indices[0]) * train_size)]
        test_index = indices[0][int(len(indices[0]) * train_size):]

        # Extract features and labels for training set
        train_feat1 = data['Features1'][train_index]
        train_feat2 = data['Features2'][train_index]
        train_feat3 = data['Features3'][train_index]
        train_feat4 = data['Features4'][train_index]
        train_feat5 = data['Features5'][train_index]
        train_feat6 = data['Features6'][train_index]
        train_feat7 = data['Features7'][train_index]
        train_feat8 = data['Features8'][train_index]
        train_lab = labels[train_index]

        # Extract features and labels for testing set
        test_feat1 = data['Features1'][test_index]
        test_feat2 = data['Features2'][test_index]
        test_feat3 = data['Features3'][test_index]
        test_feat4 = data['Features4'][test_index]
        test_feat5 = data['Features5'][test_index]
        test_feat6 = data['Features6'][test_index]
        test_feat7 = data['Features7'][test_index]
        test_feat8 = data['Features8'][test_index]

        test_lab = labels[test_index]

        # Extend the lists with the current class data
        x_train1.extend(train_feat1)
        x_train2.extend(train_feat2)
        x_train3.extend(train_feat3)
        x_train4.extend(train_feat4)
        x_train5.extend(train_feat5)
        x_train6.extend(train_feat6)
        x_train7.extend(train_feat7)
        x_train8.extend(train_feat8)

        y_train.extend(train_lab)

        x_test1.extend(test_feat1)
        x_test2.extend(test_feat2)
        x_test3.extend(test_feat3)
        x_test4.extend(test_feat4)
        x_test5.extend(test_feat5)
        x_test6.extend(test_feat6)
        x_test7.extend(test_feat7)
        x_test8.extend(test_feat8)

        y_test.extend(test_lab)

    # Convert the lists to numpy arrays
    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    x_train3 = np.array(x_train3)
    x_train4 = np.array(x_train4)
    x_train5 = np.array(x_train5)
    x_train6 = np.array(x_train6)
    x_train7 = np.array(x_train7)
    x_train8 = np.array(x_train8)

    y_train = np.array(y_train)

    x_test1 = np.array(x_test1)
    x_test2 = np.array(x_test2)
    x_test3 = np.array(x_test3)
    x_test4 = np.array(x_test4)
    x_test5 = np.array(x_test5)
    x_test6 = np.array(x_test6)
    x_test7 = np.array(x_test7)
    x_test8 = np.array(x_test8)

    y_test = np.array(y_test)

    train_samples = x_train1.shape[0]
    train_indices = np.random.permutation(train_samples)

    x_train1 = x_train1[train_indices]
    x_train2 = x_train2[train_indices]
    x_train3 = x_train3[train_indices]
    x_train4 = x_train4[train_indices]
    x_train5 = x_train5[train_indices]
    x_train6 = x_train6[train_indices]
    x_train7 = x_train7[train_indices]
    x_train8 = x_train8[train_indices]


    y_train = y_train[train_indices]

    test_samples = x_test1.shape[0]
    test_indices = np.random.permutation(test_samples)

    x_test1 = x_test1[test_indices]
    x_test2 = x_test2[test_indices]
    x_test3 = x_test3[test_indices]
    x_test4 = x_test4[test_indices]
    x_test5 = x_test5[test_indices]
    x_test6 = x_test6[test_indices]
    x_test7 = x_test7[test_indices]
    x_test8 = x_test8[test_indices]

    y_test = y_test[test_indices]

    # Separate features and labels after shuffling
    return [x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8,
            x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8,
            y_train, y_test]


class Analysis:

    def __init__(self, data, dataset):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.dataset = dataset
        self.epochs = 500
        self.perf_epochs = [100, 200, 300, 400, 500]

    def ComparativeAnalysis(self):
        """
        Perform Comparative Analysis to compare the proposed method with existing methods.

        Vary the training percentage and use different classification methods.

        Save the results in numpy files for each method and training percentage.
        """
        # Initialize lists to store comparative analysis results
        ComparativeResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Comparative Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)



            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5], 'x_train7': data[6], 'x_train8': data[7],
                      'x_test1': data[8], 'x_test2': data[9], 'x_test3': data[10], 'x_test4': data[11],
                      'x_test5': data[12], 'x_test6': data[13], 'x_test7': data[14], 'x_test8': data[15],
                      'y_train': data[16], 'y_test': data[17], 'epochs': self.epochs}



            Ne = Network(**params)


            # Perform cl classification using different methods and get predictions
            output = [
                # Ne.LIME(),
                # Ne.CLSTM(),
                # Ne.BTDN(),
                # Ne.ELM(),
                # Ne.NAS(),
                # Ne.SAlexNet(),
                # Ne.Inception(),
                Ne.HFACNet()
            ]

            # Calculating the Performance
            ComparativeResults.append([Evaluation_Metrics(data[17], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A']
        # , 'L', 'M'
        file_names = [f'Analysis1\\Dataset{self.dataset}\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults])
        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')


    def PerformanceAnalysis(self):
        """
        Perform Performance Analysis to check the maximum performance of the proposed method.

        Vary the training percentage and epochs.

        Save the results in numpy files for each training percentage and epoch combination.
        """
        # Initialize lists to store performance analysis results
        PerformanceResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)


            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5], 'x_train7': data[6], 'x_train8': data[7],
                      'x_test1': data[8], 'x_test2': data[9], 'x_test3': data[10], 'x_test4': data[11],
                      'x_test5': data[12], 'x_test6': data[13], 'x_test7': data[14], 'x_test8': data[15],
                      'y_train': data[16], 'y_test': data[17], 'epochs': self.epochs}


            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.HFACNet(epochs=self.perf_epochs[0]),
                Ne.HFACNet(epochs=self.perf_epochs[1]),
                Ne.HFACNet(epochs=self.perf_epochs[2]),
                Ne.HFACNet(epochs=self.perf_epochs[3]),
                Ne.HFACNet(epochs=self.perf_epochs[4])]

            # Calculating the Performance
            PerformanceResults.append([Evaluation_Metrics(data[17], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E']
        # , 'L', 'M'
        file_names = [f'Analysis1\\Dataset{self.dataset}\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults])


        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')

    def ROCAnalysis(self):
        print(colored("[INFO] Executing ROC Analysis", 'grey', on_color='on_white'))

        TPR = []
        FPR = []

        # Define a list of training set percentages.
        Tr_Per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for i in range(len(Tr_Per)):
            print(colored("Analysis Count Is {0} Out Of 9".format(i + 1), 'blue', on_color='on_grey'))

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=Tr_Per[i])



            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5], 'x_train7': data[6], 'x_train8': data[7],
                      'x_test1': data[8], 'x_test2': data[9], 'x_test3': data[10], 'x_test4': data[11],
                      'x_test5': data[12], 'x_test6': data[13], 'x_test7': data[14], 'x_test8': data[15],
                      'y_train': data[16], 'y_test': data[17], 'epochs': self.epochs}

            Ne = Network(**params)



            # Perform cl classification using different methods and get predictions
            # Perform cl classification using different methods and get predictions
            pred_c1 = Ne.HFACNet()



            # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for each classifier.
            [TPR1c, FPR1c] = Evaluation_Metrics1(data[17], pred_c1)


            # Store FPR and TPR for all classifiers.
            TPR_all = [TPR1c]
            FPR_all = [FPR1c]

            TPR.append(TPR_all)
            FPR.append(FPR_all)

        np.save(f'{os.getcwd()}\\Analysis1\\Dataset{self.dataset}\\TPR.npy', TPR)
        np.save(f'{os.getcwd()}\\Analysis1\\Dataset{self.dataset}\\FPR.npy', FPR)


        print(colored("[INFO] ROC Analysis Completed", 'green', on_color='on_grey'))
