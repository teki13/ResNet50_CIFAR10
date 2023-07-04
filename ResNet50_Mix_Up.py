import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time

#a function that calculates the loss of the model when using mix up
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
    def __init__(self, alpha=1.0, sampling_method=1, use_cuda=True):
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.sampling_method = sampling_method

    def __call__(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.sampling_method == 1:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.random.uniform(0.8, 1.0)

        batch_size = x.size()[0]
        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

def inference(model_1, testloader, classes, model_name):
    '''
    This function performs inference on the model
    Inputs: model_1 - the model
            testloader - the test set
            classes - the classes of the dataset (list)
            model_name - the name of the model (string)
    Output: None
    '''
    #iterate through the test set
    dataiter = iter(testloader)
    ## inference
    images, labels = next(dataiter)
    print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(36)))
    
    outputs = model_1(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    # save to images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    filename = model_name + "_result.png"
    im.save(filename)
    print('result.png saved.')


def train_model(trainloader, testloader, net_1, criterion_1, optimizer_1, mixup, image, model_name):
    '''
    This function trains the model
    Inputs: trainloader - the training set
            testloader - the test set
            net_1 - the model
            criterion_1 - the loss function
            optimizer_1 - the optimizer
            mixup - the mix up function of type class MixupData
            image - a boolean value that indicates if the first batch to which a mix up was performed was saved
            model_name - the name of the model (string)
    Output: net_1 the trained model
    '''
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        
            inputs, labels = data
            
            inputs, targets = inputs.to(device), labels.to(device)

            # get the inputs; data is a list of [inputs, labels]
            inputs, targets_a, targets_b, lam = mixup(inputs, targets)

            
            #if this is the first batch to which a mix up was performed, save it as a 'mixup.png' file
            if image == 0:
                print("Here")
                im = Image.fromarray((torch.cat(inputs.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
                filename = model_name + "_mixup.png"
                im.save(filename)
                image += 1


            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                        targets_a, targets_b))

            # zero the parameter gradients
            optimizer_1.zero_grad()

            # forward + backward + optimize
            outputs = net_1(inputs)
            
            #calculate the loss
            loss = mixup_criterion(criterion_1, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer_1.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


        # run the model on the test set at each epoch
        correct = 0
        total = 0
        starttime = time.time()
        with torch.no_grad():
            for batch_size, (inputs, targets) in enumerate(testloader):

                images, labels = inputs.to(device), targets.to(device)
                outputs = net_1(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        accuracy = 100 * correct / total
        print('Test accuracy for epoch %d is %d %%' % (epoch + 1, accuracy))

    return net_1

if __name__ == '__main__':

    #connect to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16
    #load the train set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size_test = 36
    #load the test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Define the ResNet50 model 1 which perfroms mix up using sampling method 1
    net_1 = torchvision.models.resnet50()
    num_classes = 10  # CIFAR10 has 10 classes
    num_ftrs = net_1.fc.in_features
    net_1.fc = torch.nn.Linear(num_ftrs, num_classes)

    ## loss and optimiser for the model which performs miz up using sampling methos 1
    criterion_1 = torch.nn.CrossEntropyLoss()
    optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=0.001)

    net_1.to(device)
    criterion_1.to(device)

    ## train the model using mix up with sampling method 1
    mixup = MixupData(alpha = 0.2, sampling_method = 1, use_cuda=False)
    net_1 = train_model(trainloader, testloader, net_1, criterion_1, optimizer_1, mixup, 0, 's_1')

    print('Training done.')
    # save trained model
    torch.save(net_1.state_dict(), 'saved_model_s_1.pt')
    print('Model saved.')


    #Run the trained model on the test set and save the output for 36 images

    ## load the trained model
    model_1 = torchvision.models.resnet50()
    num_ftrs_loaded = model_1.fc.in_features
    model_1.fc = torch.nn.Linear(num_ftrs_loaded, 10)
    model_1.load_state_dict(torch.load('saved_model_s_1.pt'))

    #make inference and save the output
    inference(model_1, testloader, classes, 's_1')




    #next we train the resnet model using sampling method 2
    # Define the ResNet50 model 2 which perfroms mix up using sampling method 2
    net_2 = torchvision.models.resnet50()
    num_classes = 10  # CIFAR10 has 10 classes
    num_ftrs = net_2.fc.in_features
    net_2.fc = torch.nn.Linear(num_ftrs, num_classes)

    ## loss and optimiser for the model which performs miz up using sampling method 2
    criterion_2 = torch.nn.CrossEntropyLoss()
    optimizer_2 = torch.optim.Adam(net_2.parameters(), lr=0.001)

    ## train the model using mix up with sampling method 1
    mixup = MixupData(sampling_method = 2, use_cuda=False)
    net_2 = train_model(trainloader, testloader, net_2, criterion_2, optimizer_2, mixup, 0, 's_2')

    print('Training done.')
    # save trained model
    torch.save(net_2.state_dict(), 'saved_model_s_2.pt')
    print('Model saved.')

    ## load the trained model
    model_2 = torchvision.models.resnet50()
    num_ftrs_loaded = model_2.fc.in_features
    model_2.fc = torch.nn.Linear(num_ftrs_loaded, 10)
    model_2.load_state_dict(torch.load('saved_model_s_2.pt'))

    #make inference and save the output
    inference(model_2, testloader, classes, 's_2')




