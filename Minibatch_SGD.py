import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time


def polynomial_fun(w,x):
    '''
    Input: x - a torch tensor of shape N x 1 ex. torch.tensor([[1],[2],[3]]) for N = 3
           w - a torch tensor of shape M x 1 ex. torch.tensor([[0.5], [0.2]]) for M = 2
    
    Output:
           y - a torch tensor of shape N x 1 
    '''
    M = len(w)
    y = torch.matmul(x.pow(torch.arange(M, dtype=torch.float32)),w)
    
    return y


def fit_polynomial_ls(x,t,M):
    '''
    Inputs: x - 1D torch tensor containing the different pairs of values for x
            t - 1D toerch tensor containing the target values
            M - int (scalar) containing the degree of the polynomial

    Outout: w - 1D torch tensor containing the optimum weights calculated by the
            least squares solution
            time_ls_train - float containing the time taken to calculate the weights (int)
    '''

    starttime = time.time()
    n = len(x)
    x_values = np.zeros((n, M+1))

    for pair in range(n): 
        for i in range(M+1):
            x_values[pair][i] = x[pair] ** i

    #convert to torch tensor
    x_values = torch.from_numpy(x_values).float()
    

    #calculate the least squared loss
    w = torch.linalg.lstsq(x_values, t.float(), driver = 'gels').solution
    
    time_ls_train = round((time.time() - starttime), 2)
    
    return w, time_ls_train


def fit_polynomial_sgd(x,t,M,learning_rate,minibatch):
    '''
    A function taht performs stochastic gradient descent to find the optimum weights
    for a polynomial of degree M
    Inputs: x - 1D torch tensor containing the different pairs of values for x
            t - 1D toerch tensor containing the target values
            M - int (scalar) containing the degree of the polynomial
            learning_rate - float containing the learning rate
    Outout: w - 1D torch tensor containing the optimum weights calculated by the
            SGD solution
            time_sgd_train - float containing the time taken to calculate the weights (int)
    '''
    #start the timer
    starttime = time.time()


    #initilaize the weights with random numbers
    w = torch.randn((M+1, 1), dtype=torch.float32, requires_grad=True)

    #define an optimizer
    optimizer = torch.optim.Adam([w], lr=learning_rate)

    dataset = TensorDataset(x, t)
    dataloader = DataLoader(dataset, batch_size=minibatch, shuffle=True)
    

    for epoch in range(100000):
        
        for i, (batch_x, batch_t) in enumerate(dataloader):

            #calculate the prediction
            y_pred = polynomial_fun(w, batch_x)
            
            #calculate the squared loss
            loss = torch.mean((y_pred - batch_t)**2)
            
            optimizer.zero_grad()
            loss.backward()

            #update the weights
            optimizer.step()
        
        if epoch % 1000 == 0:


            #y_pred = torch.tensor(y_pred, dtype=torch.float32)
            y_pred = polynomial_fun(w,x)
            loss = torch.mean((y_pred - t)**2)

            print(f"Epoch {epoch}: Loss = {loss.item()}")
        
        if epoch == 40000:
            #decrease the learning rate
            optimizer.param_groups[0]['lr']  = 0.0001
        
        if epoch == 80000:
            #decrease the learning rate
            optimizer.param_groups[0]['lr']  = 0.00001
    
    #end the timer
    time_sgd_train = round((time.time() - starttime), 2)

    return w.detach(), time_sgd_train


if __name__ == '__main__':

    #define the weight
    w = torch.FloatTensor([[1],[2],[3],[4],[5]])
    w_t = torch.FloatTensor([[0],[1],[2],[3],[4],[5]])


    #generate train data
    x_train = torch.FloatTensor(100, 1).uniform_(-20, 20)
    y = polynomial_fun(w,x_train)
    y_train = y + torch.randn(y.size()) * 0.2


    #generate the test data
    x_test = torch.FloatTensor(50, 1).uniform_(-20, 20)
    y_t = polynomial_fun(w,x_test)
    y_test = y_t + torch.randn(y_t.size()) * 0.2


    #Find the weights using the least squares solution
    w_pred, time_ls_train = fit_polynomial_ls(x_train,y_train,5)

    #Predict the y for both the train and test set
    y_pred_train = polynomial_fun(w_pred,x_train)
    y_pred_test = polynomial_fun(w_pred,x_test)

    #Print out the observed data versus the true polynomial data
    difference = y_train - y
    print("The mean of the difference between the observed training data and the true ploynomial curve is ", torch.mean(difference).item())
    print("The standard deviation of the difference between the observed training data and the true ploynomial curve is ", torch.std(difference).item())
    difference_predicted = y_pred_train - y
    print("The mean of the difference between the LS predicted train values and the true values is ", difference_predicted.mean().item())
    print("The standard deviation of the difference between the LS predicted train values and the true values is ", torch.std(difference_predicted).item())
    difference_test = y_pred_test - y_t
    print("The mean of the difference between the LS predicted test values and the true values is ", difference_test.mean().item())
    print("The standard deviation of the difference between the LS predicted test values and the true values is ", torch.std(difference_test).item())

    #Predict the weights using SGD
    w_pred_sgd, time_sgd_train = fit_polynomial_sgd(x_train,y_train,5,0.001, 50)

    #make predictions with the test and train data using w predicted by sgd
    y_pred_train_sgd = polynomial_fun(w_pred_sgd,x_train)
    y_pred_test_sgd = polynomial_fun(w_pred_sgd,x_test)

    difference = y_pred_train_sgd - y
    print("The mean of the difference between the SGD predicted train values and the true values is ", torch.mean(difference).item())
    print("The standard deviation of the difference between the SGD predicted train values and the true values is ", torch.std(difference).item())
    difference_test = y_pred_test_sgd - y_t
    print("The mean of the difference between the SGD predicted test values and the true values is ", torch.mean(difference_test).item())
    print("The standard deviation of the difference between the SGD test predicted and the true values is ", torch.std(difference_test).item())

    #Compute the mean square error
    least_square_error = torch.sqrt(torch.mean((y_pred_test-y_t)**2))
    sgd_rmse = torch.sqrt(torch.mean((y_pred_test_sgd-y_t)**2))
    print("RMSE LS for the value of y is ", least_square_error.item())
    print("RMSE SGD for the value of y is ", sgd_rmse.item())

    least_square_error_w = torch.sqrt(torch.mean((w_pred-w_t)**2))
    sgd_rmse_w = torch.sqrt(torch.mean((w_pred_sgd-w_t)**2))
    print("The RMSE for w for the least squre error method is ", least_square_error_w.item())
    print("The RMSE for w for the SGD method is ", sgd_rmse_w.item())

    #report the speed fo predictions using least squares and sgd
    print("The total time it took to train w using LS in seconds is ", time_ls_train)
    print("The total time it took to train w using SGD in seconds is ", time_sgd_train)



