import math
import numpy as np

class Neuron:
    def __init__(self, layer, index, n_weights):
        self.layer = layer
        self.index = index
        self.weights = np.random.uniform(low = -0.5, high = 0.5, size = [1, n_weights])[0]
        self.theta = np.random.uniform(low = -0.5, high = 0.5)

    def __str__(self):
        parsed_neuron = "\nNeuron " + str(self.index) + " " + self.layer + " layer" + ":\n"
        i = 0

        for w in self.weights:
            parsed_neuron += "w_" + self.layer + "_" + str(self.index) + str(i) + " = \t" + str(w) + "\n"
            i += 1

        parsed_neuron += "theta_" + self.layer + "_" + str(self.index) + " =\t" + str(self.theta) + "\n"

        return parsed_neuron

    def __repr__(self):
        return str(self)

    def as_vector(self):
        return np.append(self.weights, self.theta)

class Layer:
    def __init__(self, layer, n_neurons):
        self.neurons = [Neuron(layer, i, n_neurons) for i in range(n_neurons)]

    def __iter__(self):
        return iter(self.neurons)

    # def __next__(self):
    #     if self.current > self.high:
    #         raise StopIteration
    #     else:
    #         self.current += 1
    #         return self.current - 1

    def __str__(self):
        parsed_neuron = ""

        for neuron in self.neurons:
            parsed_neuron += str(neuron)

        return parsed_neuron

    def as_matrix(self):
        return np.matrix([neuron.as_vector() for neuron in self.neurons])

class Model:
    def __init__(self, i_neurons = 2, h_neurons = 2, o_neurons = 1):
        self.i_neurons = i_neurons
        self.h_neurons = h_neurons
        self.o_neurons = o_neurons

        self.h_layer = Layer("h", h_neurons)
        self.o_layer = Layer("o", o_neurons)

    def f(self, net):
        return (1/(1+(math.exp(-net))))

    def df(self, net):
        return (self.f(net) * (1-self.f(net)))

    def forward(self, model, x_p):
        # Compute net and f(net) for every neuron on the hidden layer
        f_h_net_h_pj = np.zeros(self.h_neurons)
        df_h_net_h_pj = np.zeros(self.h_neurons)
        for i, neuron in enumerate(self.h_layer):
            net_h_pj = np.dot(np.append(x_p, 1), neuron.as_vector())
            f_h_net_h_pj[i] = model.f(net_h_pj)
            df_h_net_h_pj[i] = model.df(net_h_pj)

        # Compute net and f(net) for every neuron on the output layer
        f_o_net_o_pj = np.zeros(self.o_neurons)
        df_o_net_o_pj = np.zeros(self.o_neurons)
        for i, neuron in enumerate(self.o_layer):
            net_o_pj = np.dot(np.append(f_h_net_h_pj, 1), neuron.as_vector())
            f_o_net_o_pj[i] = model.f(net_o_pj)
            df_o_net_o_pj[i] = model.df(net_o_pj)

        return f_h_net_h_pj, df_h_net_h_pj, f_o_net_o_pj, df_o_net_o_pj

tst = Model(2)
tst.forward(tst, [1,1])
