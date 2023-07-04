import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
import numpy as np
from collections import Counter
import time
from network_pt import Net


#function to calculate the loss for the mix-up train data
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    This function calculates the loss of the model when using mix up
    Inputs: criterion - the loss function
            pred - the prediction of the model of type torch.tensor size (batch_size, 1)
            y_a - the first target labels of type torch.tensor size (batch_size, 1)
            y_b - the second target labels of type torch.tensor size (batch_size, 1)
            lam - the lambda value (int)
    Output: the loss of the model (int)
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#function to calculate f1 score, return the percision, recall and f1 score for each of the classes
def f1_score(confusion_matrix):
    '''
    This function calculates the f1 score for each of the classes
    Inputs: confusion_matrix - the confusion matrix of the model of type np.array size (num_classes, num_classes)
    Output: percision - the percision of the model for each of the classes of type np.array size (num_classes)
            recall - the recall of the model for each of the classes of type np.array size (num_classes)
            f1_score_res - the f1 score of the model for each of the classes of type np.array size (num_classes)
    '''
    
    #take all the true positive values
    TP = confusion_matrix.diagonal()

    #sum of True Positives and False Negatives
    TP_FN = np.sum(confusion_matrix, axis = 1)
    #edge case - Nan could appear in the case of the model being trained without a certain class
    #and prediction would not be made with this class
    TP_FN = np.nan_to_num(TP_FN, nan = 0)

    #sum of True Positives and False Positives
    TP_FP = np.sum(confusion_matrix, axis = 0)
    #edge case
    TP_FP = np.nan_to_num(TP_FP, nan = 0)

    #calculate percision
    percision = TP / TP_FP

    #calculate recall 
    recall = TP / TP_FN

    #calculate F1 score
    f1_score_res = 2 * (percision * recall) / (percision + recall)
    f1_score_res = np.nan_to_num(f1_score_res, nan = 0)

    return percision, recall, f1_score_res


#Class that performs mix-up
class MixupData:
    '''
    A class that performs mix up of x and y
    Inputs: alpha - the alpha value (int)
            sampling_method - the sampling method (int)
            use_cuda - a boolean value that indicates if cuda is used
            x - the input data of type torch.tensor size (batch_size, 1)
            y - the target labels of type torch.tensor size (batch_size, 1)
    Output: mixed_x - the mixed input data of type torch.tensor size (batch_size, 1)
            y_a - the first target labels of type torch.tensor size (batch_size, 1)
            y_b - the second target labels of type torch.tensor size (batch_size, 1)
            lam - the lambda value (int)
    '''
    def __init__(self, alpha=1.0, sampling_method=1):
        self.alpha = alpha
        self.sampling_method = sampling_method

    def __call__(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.sampling_method == 1:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.random.uniform(0.8, 1.0)

        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


#evaluate model and record summary statistics
def evaluate_model(criterion, net, dataset):

    '''
    This function evaluates the model and calculates the accuracy, percision, recall, f1 score and loss
    Inputs: criterion - the loss function
            net - the model
            dataset - the dataset of type torch.utils.data.DataLoader
    Output: accuracy - the accuracy of the model (int)
            percision - the percision of the model for each of the classes of type np.array size (num_classes)
            recall - the recall of the model for each of the classes of type np.array size (num_classes)
            f1_score_res - the f1 score of the model for each of the classes of type np.array size (num_classes)
            loss_value - the loss of the model (int)
    '''

    #initialize a confusion matrix
    confusion_matrix = np.zeros((10,10))

    #initialize variables
    running_loss = 0.0
    total = 0.0
    correct = 0.0

    for batch_size, (inputs, targets) in enumerate(dataset):
        
        images, targets = inputs.to(device), targets.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss_predicted = criterion(outputs, targets)
        running_loss += loss_predicted.item()
        total += targets.size(0)
        correct += (predicted == targets).sum().item() #true positives
        #update the confusion matrix
        for i in range(len(targets)):
            confusion_matrix[targets[i], predicted[i]] += 1

    #calculate the accuracy
    accuracy = 100 * correct / total

    #calculate the percision, recall and f1 score
    percision, recall, f1_score_res = f1_score(confusion_matrix)

    #calculate the loss
    loss_value = running_loss/len(dataset)

    return accuracy, percision, recall, f1_score_res, loss_value



#a function for printing the summary statistics
def print_summary_statistics(accuracy, percision, recall, f1score, loss, type_set, speed):
        '''
        This function prints the summary statistics
        Inputs: accuracy - the accuracy of the model (int)
                percision - the percision of the model for each of the classes of type np.array size (num_classes)
                recall - the recall of the model for each of the classes of type np.array size (num_classes)
                f1score - the f1 score of the model for each of the classes of type np.array size (num_classes)
                loss - the loss of the model (int)
                type_set - the type of the set (str)
                speed - the speed of the model for each of the batches of type list
        Output: None
        '''
        sum_speed = sum(speed)
        mean_speed = sum_speed / len(speed)
        print("Summary Statistics for ",type_set, "Set")
        print("-" * 95)
        print(f"{'Accuracy':<15}{'Val.Loss':<15}{'Total Speed (sec)':<25}{'Avg. Speed(sec)':<25}")
        print("-" * 95)
        print(f"{round(accuracy,3):<15}{round(loss,3):<15}{round(sum_speed,3):<25}{round(mean_speed,3):<25}")
        print("-" * 95)
        print(f"{'Class':<25}{'Precision':<25}{'Recall':<25}{'F1 Score':<20}")
        print("-" * 95)
        
        for i in range(len(percision)):
          print(f"{i:<25}{round(percision[i],3):<25}{round(recall[i],3):<25}{round(f1score[i],3):<20}")
        print("-" * 95)


#function that is used to train the model
def train_model(net, optimizer, criterion, dev_loader, val_loader, mixup):
    '''
    Thus function trains the model
    Inputs: net - the model
            optimizer - the optimizer
            criterion - the loss function
            dev_loader - the train dataset of type torch.utils.data.DataLoader
            val_loader - the validation dataset of type torch.utils.data.DataLoader
            mixup - a boolean value that indicates if mixup is used or not
    Output: None
    '''
    
    #lists to save the train and validation losses
    loss_values_train = []
    loss_values_test = []

    #lists to save the speed of the train and validation loss
    speed_test = []
    speed_train = []

    #lists to save the test and validation metric values
    metric_test = []
    percision_test = []
    recall_test = []
    f1_score_test = []
    
    # loop over the dataset multiple times
    for epoch in range(10):  

        starttime = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        print("Epoch", epoch)
        for i, data in enumerate(dev_loader, 0):
        
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets_a, targets_b, lam = mixup(inputs, labels)


            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                        targets_a, targets_b))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            #calculate the loss
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            # cumulative loss
            running_loss += loss.item()


        loss_values_train.append(running_loss/len(dev_loader))
        #stop the timer
        totaltime = round((time.time() - starttime), 2)
        speed_train.append(totaltime)
        

        # run the model on the test set at each epoch
        correct = 0
        total = 0
        running_loss = 0.0
        starttime = time.time()

        #initialize a confusion matrix
        confusion_matrix = np.zeros((10,10))

        
        with torch.no_grad():
            for batch_size, (inputs, targets) in enumerate(val_loader):
                
                images, targets = inputs.to(device), targets.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                loss_predicted = criterion(outputs, targets)
                running_loss += loss_predicted.item()
                total += targets.size(0)
                correct += (predicted == targets).sum().item() #true positives
                
                #update the confusion matrix
                for i in range(len(targets)):
                    confusion_matrix[targets[i], predicted[i]] += 1

                
        loss_val = running_loss/len(val_loader)
        loss_values_test.append(loss_val)
        totaltime = round((time.time() - starttime), 2)
        speed_test.append(totaltime)
        

        #calculate and save accuracy
        accuracy = 100 * correct / total
        metric_test.append(accuracy)
        

        #calculate and save percision, recall and f1 score
        percision, recall, f1_score_res = f1_score(confusion_matrix)
    
        percision_test.append(percision)
        recall_test.append(recall_test)
        f1_score_test.append(f1_score_res)
        

        print(f"{'Epoch':<10}{'Accuracy':<15}{'Val.Loss':<15}")
        print("-" * 95)
        print(f"{epoch:<10}{round(accuracy,3):<15}{round(loss_val,2):<15}")
        print("-" * 95)
        print(f"{'Class':<25}{'Precision':<25}{'Recall':<25}{'F1 Score':<20}")
        print("-" * 95)
        
        for i in range(len(percision)):
          print(f"{i:<25}{round(percision[i],3):<25}{round(recall[i],3):<25}{round(f1_score_res[i],3):<20}")
        print("-" * 95)

    
    return net, loss_values_train, loss_values_test, speed_test, speed_train, metric_test, percision_test, recall_test,f1_score_test




if __name__ == '__main__':
    
    #check weather gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #load the train and test set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    #concetanate the test and train set into one set
    cifar_combined = torch.utils.data.ConcatDataset([trainset, testset])

    #split the dataset into 80% train and 20% test
    train_size = int(0.8 * len(cifar_combined))
    test_size = len(cifar_combined) - train_size

    #furter split the train into 90 % development and 10% validation set
    dev_size = int(0.9 * train_size)
    val_size = train_size - dev_size

    dev_dataset, val_dataset, test_dataset = torch.utils.data.random_split(cifar_combined, [dev_size, val_size, test_size])

    #prepare the data loader for the dev, val, and test sets
    batch_size = 16
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    #define the classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(dev_loader)
    images, labels = next(dataiter)

    # Define the first model
    net_1 = Net().to(device)
    ## loss and optimiser for the model which performs mix up using sampling method 1 and optimizer Adam
    criterion_1 = torch.nn.CrossEntropyLoss().to(device)
    optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=0.001)

    #train the first model
    mixup = MixupData(alpha = 0.2, sampling_method = 1)
    net, loss_values_train, loss_values_test, speed_test, speed_train, metric_test, percision_test, recall_test,f1_score_test = train_model(net_1, optimizer_1, criterion_1, dev_loader, val_loader, mixup)

    # save the first model
    torch.save(net.state_dict(), 'adam_model.pt')
    print('Model saved.')


    #load model 1
    model_1 = Net().to(device)
    model_1.load_state_dict(torch.load('adam_model.pt'))

    #apply function on train, validation, and test set to obtain summary statistics
    accuracy_d, percision_d, recall_d, f1_score_d, loss_value_d = evaluate_model(criterion_1, model_1, dev_loader)
    accuracy_v, percision_v, recall_v, f1_score_v, loss_value_v = evaluate_model(criterion_1, model_1, val_loader)
    accuracy_h, percision_h, recall_h, f1_score_h, loss_value_h = evaluate_model(criterion_1, model_1, test_loader)

    print_summary_statistics(accuracy_d, percision_d, recall_d, f1_score_d, loss_value_d , 'Development', speed_train)
    print_summary_statistics(accuracy_v, percision_v, recall_v, f1_score_v, loss_value_v , 'Validation', speed_test)
    print_summary_statistics(accuracy_h, percision_h, recall_h, f1_score_h, loss_value_h , 'Holdout', [0]) #i put 0 for speed in this case because we were not asked to compute this speed



    #define the second model
    net_2 = Net().to(device)
    ## loss and optimiser for the model which performs mix up using sampling method 1 and optimizer SGD
    criterion_2 = torch.nn.CrossEntropyLoss().to(device)
    optimizer_2 = optim.SGD(net_2.parameters(), lr=0.001, momentum=0.9)

    #train the second model
    mixup = MixupData(alpha = 0.2, sampling_method = 1)
    net, loss_values_train, loss_values_test, speed_test, speed_train, metric_test, percision_test, recall_test,f1_score_test = train_model(net_2, optimizer_2, criterion_2, dev_loader, val_loader, mixup)

    # save the second model
    torch.save(net.state_dict(), 'sgd_model.pt')
    print('Model saved.')

    #load model 2
    model_2 = Net().to(device)
    model_2.load_state_dict(torch.load('sgd_model.pt'))

    #apply function on train, validation, and test set to obtain summary statistics
    accuracy_d, percision_d, recall_d, f1_score_d, loss_value_d = evaluate_model(criterion_2, model_2, dev_loader)
    accuracy_v, percision_v, recall_v, f1_score_v, loss_value_v = evaluate_model(criterion_2, model_2, val_loader)
    accuracy_h, percision_h, recall_h, f1_score_h, loss_value_h = evaluate_model(criterion_2, model_2, test_loader)

    print_summary_statistics(accuracy_d, percision_d, recall_d, f1_score_d, loss_value_d , 'Development', speed_train)
    print_summary_statistics(accuracy_v, percision_v, recall_v, f1_score_v, loss_value_v , 'Validation', speed_test)
    print_summary_statistics(accuracy_h, percision_h, recall_h, f1_score_h, loss_value_h , 'Holdout', [0])
