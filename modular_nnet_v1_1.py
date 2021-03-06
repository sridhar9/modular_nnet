import numpy as np
import random
import gzip
import pickle
import matplotlib.pyplot as plt 
import time

temp_var = ""

def main ():
    path = "C:\\Users\\sridhar\\source\\repos\\ModularNnetwork\\ModularNnetwork\\data\\mnist.pkl.gz"
    training_data, val_data, test_data = Utils.get_data_wrapper(path)

    # with below configuration, This code was able to achieve 95.5% test accuracy on MNIST handwritten digits recog task.
    # batch_size= 10; epochs=50; sizes=[784,30,10]; eta=0.01;
    
    batch_size = 10
    epochs = 50                 # This number of times we iterate on the whole training data.
    sizes = [784,30,10]
    eta = 0.01                  
    net = Network(sizes)
    st_time = time.time()
    cost_list = net.stochastic_gd(training_data, epochs, eta, batch_size, test_data)
    end_time = time.time()


    title = "Network:"+str(net.sizes)+" eta: "+str(eta)+" batch_size:"+str(batch_size)+" epochs:"+str(epochs)
    Utils.plot_graph(cost_list, title, temp_var)





class Network:

    # Features of this L-layer neural network implementation:
    # 1. fully vectorized(whole mini-batch of training examples processed at once)
    # 2. stochostic gradient descent implemented
    # 3. fully modularized implementation
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((i,1)) for i in self.sizes[1:]]         # This way of doing is called "list-comprehension"
        self.weights = [np.random.randn(j,i)/np.sqrt(i) for i,j in zip(self.sizes[:-1], self.sizes[1:])]        # np.sqrt(i) => solves exploding/vanishing gradient descent problem.
        self.act_func = "sigmoid"
        self.cost_func = "cross_entropy"

    def stochastic_gd(self, training_data, epochs, eta, mini_batch_size, test_data=None):
        # This function is responsible for stochasticness and mini-batches.
        cost_list = []
        for iter in range (epochs):
            cost_minibatch = 0
            random.shuffle (training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                cost_minibatch += self.update_minibatch(mini_batch, eta)
            avg_cost_minibatch = (1.0*cost_minibatch)/len(training_data)
            if (test_data):
                count, total, avg_cost = self.evaluate(test_data)
                print ("Epoch: "+str(iter)+" count: "+str(count)+" total: "+str(total)+" avg_cost: "+str(avg_cost))
                global temp_var
                temp_var = str(count)+"_"+str(total)+"_"+str(round(avg_cost,4))+".png"
            cost_list.append(avg_cost_minibatch)
        return cost_list


    def update_minibatch(self, mini_batch, eta):
        # We apply our gradient descent on this mini-batch i.e. network takes one step after completing the mini-batch. 
        # In this way, we are taking faster steps (batch size is less) and reaching the global minima faster than batch gradient descent.
        train_x, train_y = Utils.vectorize_data(mini_batch)
        # dim(train_x) = (nx, m) where nx => number of input units; m=> mini_batch_size
        # dim(train_y) = (nL, m) | nL = units of output layer.
        m = train_x.shape[1]
        Zall, Aall, cost_minibatch = self.forward_prop_vec(train_x, train_y)

        nb, nw = self.backward_prop_vec(train_y, Zall, Aall)
        # update b, w
        self.biases = [nb_temp - ((1.0*eta)*nbv)/m for nb_temp, nbv in zip (self.biases, nb)]       # divide by m because its not being done in backprop
        self.weights = [nw_temp - ((1.0*eta)*nwv)/m for nw_temp, nwv in zip (self.weights, nw)]
        return cost_minibatch

    def forward_prop_vec(self, train_x, train_y):
        Zall = []
        aprev = train_x
        Aall = []
        Aall.append(train_x)            # dim(Zall) = dim(Aall) and dim(Aall) = [[nx, m] [n1, m] [n2, m]....[nL, m]]
        for layer in range(0, self.num_layers -1):
            zl = np.dot(self.weights[layer],aprev) + self.biases[layer]
            Zall.append(zl)
            al = self.gz(zl)
            aprev = al
            Aall.append(al)
        # cost of this entire mini_batch
        ycap = Aall[-1]
        cost_minibatch = self.get_cost(ycap, train_y)
        return Zall, Aall, cost_minibatch



    def backward_prop_vec(self, train_y, Zall, Aall):
        # nb, nw are change in biases, weights when we change cost. which is same as partial derivative of cost w.r.t b, w.
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        # first let's find dbL, dwL (partial derivative of cost w.r.t bL, wL)
        delta = np.multiply(self.get_daL(self.cost_func, Aall[-1], train_y), self.gz_prime(Zall[-1]))
        # dim(delta) = (nL, m)
        nb[-1] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
        nw[-1] = np.dot(delta, Aall[-2].transpose())
        for layer in range(2,self.num_layers):
            delta = np.dot (self.weights[-layer+1].transpose(), delta) * (self.gz_prime(Zall[-layer]))
            nb[-layer] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
            nw[-layer] = np.dot (delta, Aall[-layer-1].transpose())
        return nb, nw

    def evaluate(self, test_data):
        # dim(test_data) = [((nx,1), (ny,1)), (), (m times)]
        count=0
        test_x, test_y = Utils.vectorize_data(test_data)
        # dim(test_x) = (nx, m)
        # dim(test_y) = (ny, m)
        Zall, Aall, cost_ = self.forward_prop_vec(test_x, test_y)
        ycap = Aall[-1]
        # dim(ycap) = (ny, m)
        ycap_max_ind = np.argmax(ycap, axis=0)
        y_max_ind = np.argmax(test_y, axis=0)
        for yc, y in zip(ycap_max_ind, y_max_ind):
            if (yc==y):
                count+=1
        return (count, test_x.shape[1], cost_/test_x.shape[1])


    def gz(self,zl):
        # dim(zl) = (nl, m)
        # gz => activation function which is specific to the problem. we are making it modular, so that we can use whatever function we need.
        if (self.act_func == "sigmoid"):
            al = 1.0/(1.0 + np.exp(-1.0 * zl))
        elif (self.act_func == "tanh"):
            al = (np.exp(zl) - np.exp(-1.0 * zl))/(np.exp(zl) + np.exp(-1.0 * zl))
        elif (self.act_func == "relu"):
            # TODO: following doesn't work; it is also in gz_prime.
            a_list = list(map(lambda x: 0 if (x<=0) else x, zl))
            al = np.asarray(a_list).reshape(len(a_list),1)      # converting list to numpy vector of dim (nl, 1)
        return al

    def gz_prime(self, zl):
        # partial derivative of 'al' w.r.t 'zl'
        act_func = self.act_func
        if (act_func == "sigmoid"):
            gzp = self.gz(zl) * (1 - self.gz(zl))
        elif (act_func == "tanh"):
            gzp = 1.0 - np.power(self.gz(zl), 2)
        elif (act_func == "relu"):
            a_list = list(map(lambda x: 0 if (x<=0) else 1, zl))
            gzp = np.asarray(a_list).reshape(len(a_list),1) 
        return gzp

    def get_cost(self, ycap, y):
        # dim(ycap) = dim(y) = (nL, m)
        if (self.cost_func == "cross_entropy"):
            loss_vec = -1.0 * (y * np.log(ycap) + (1-y)*np.log(1-ycap))
            loss_ = np.squeeze(np.sum(loss_vec)) 
        return loss_

    def get_daL(self, cost_func, aL, y):
        # daL => partial derivative of Cost w.r.t aL. So the dim are same as (aL)
        if (cost_func == "cross_entropy"):
            daL = np.divide((aL - y), aL*(1.0 - aL) +  pow(10,-9))      # a small value is added to denominator. so that divide by zero wont happen.
        return daL







class Utils:

    @classmethod
    def get_data_wrapper(cls, path):
        '''
        return: train_data = [(vec(784, 1), vec(10, 1)), (), ()]
        '''
        tr_data, val, tst_data = Utils.get_data(path)
        # print (np.shape(tr_data[0][:10,:]))
        train_input = [np.reshape(x, (784, 1)) for x in tr_data[0]]
        train_label = [Utils.vectorize_ylayer(y) for y in tr_data[1]]
        train_data = list(zip(train_input, train_label))

        valid_input = [np.reshape(x, (784, 1)) for x in val[0]]
        valid_label = [Utils.vectorize_ylayer(y) for y in val[1]]
        validation_data = list(zip(valid_input, valid_label))

        test_input = [np.reshape(x, (784, 1)) for x in tst_data[0]]
        test_label = [Utils.vectorize_ylayer(y) for y in tst_data[1]]
        test_data = list(zip(test_input, test_label))
        return (train_data, validation_data, test_data)

    @classmethod
    def vectorize_ylayer(cls, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @classmethod
    def get_data(cls, path):
        '''
        :param data_path:
        :return:    training_data = [v(748 x 50000), v(1 x 50000)]
                    validation_data = [v(748 x 10000), v(1 x 10000)]
                    test_data = [v(748 x 10000), v(1 x 10000)]
        '''
        f = gzip.open(path, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        f.close()
        return training_data, validation_data, test_data

    @classmethod
    def vectorize_data(cls, data):
        '''
        input: [(vec(nx,1), vec(ny,1)), (), (m times)]
        output: [vec(nx, m), vec(ny, m)]
        '''
        m = len(data)
        nx, t = data[0][0].shape
        ny, t = data[0][1].shape
        train_x = np.zeros((m, nx))
        train_y = np.zeros((m, ny))
        for i in range(m):
            train_x[i] = data[i][0].transpose()
            train_y[i] = data[i][1].transpose()
        train_x = train_x.transpose()
        train_y = train_y.transpose()
        return (train_x, train_y)

    @classmethod
    def plot_graph(cls, ylist, title, file_name):
        xlist = range(len(ylist))
        plt.plot(xlist, ylist)
        plt.title(title)
        plt.grid(True)
        plt.savefig(file_name)
        plt.show()
        
        



if (__name__ == "__main__"):
    main()

