import numpy as np
import prettytable as pt
from numpy import random

e = 0.1
cols = ["Iteration", "y", "err"]

def get_weights(data):
    np.savetxt("weights_lab_1.csv", data, delimiter = ',')

class NeuralNet:
    def __init__(this):

        np.random.seed(1)
        this.synaptic_weights = 2*random.random((4, 1)) - 1
        this.synaptic_weights = np.loadtxt("weights_lab_1.csv", delimiter=',')
        this.table = pt.PrettyTable()
        print("Initial weights: ")
        print(str(this.synaptic_weights))


    @staticmethod
    def sigmfunc(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigmfuncdf(x):
        return x*(1 - x)

    def training(this, training_set_inputs, training_set_outputs, training_iters):

        iterations = []
        outputs = []
        errors = []

        output = 0
        error = 0
        i = 0

        for iteration in range(training_iters):

            print("Iteration:\t", iteration + 1)
            output = this.activate(training_set_inputs)
            error = abs((training_set_outputs - output) / training_set_outputs)
            print("Error:\t", error[0][0])

            if (iteration + 1) % 10 == 0:
                iterations.append(i + 1)
                outputs.append(np.round(output[0], 6))
                errors.append(np.round(error[0], 6))

            with open("lab1part1temp.txt", "a") as f:
                f.write(str(iteration)+'\t'+str(error[0][0])+'\n')
            if error < e:
                break

            delta = this.sigmfuncdf(output) * (training_set_outputs - output)
            adjustment = np.dot(training_set_inputs.T, delta)
            print(adjustment[1][0])
            print(this.synaptic_weights[1])
            print(this.synaptic_weights)

            for i in range(0, 3):
                this.synaptic_weights[i] += adjustment[i][0]

            print("Synaptic weights:\t", this.synaptic_weights)
            i += 1

        get_weights(this.synaptic_weights)
        iterations.append(i + 1)
        outputs.append(np.round(output[0], 6))
        errors.append(np.round(error[0], 6))

        this.table.add_column(cols[0], iterations)
        this.table.add_column(cols[1], outputs)
        this.table.add_column(cols[2], errors)

        print(this.table)
        print("Weights: ", str(this.synaptic_weights))

    def activate(this, inputs):
        return this.sigmfunc(np.dot(inputs, this.synaptic_weights))

def save_result(input, output):
        with open("lab1part1res.txt", 'w') as f:
            f.write(str(input)+'\t'+str(output))

def main():

    neural_network = NeuralNet()

    training_set_inputs = np.array([[1, 7, 4, 5]])
    training_set_outputs = np.array([[0.3]]).T

    #Training regime
    neural_network.training(training_set_inputs, training_set_outputs, 1000)

    test_set = np.array([1, 7, 4, 5])
    test_set_output = neural_network.activate(test_set)
    print("Recognition mode: ")
    print("Initial vector: ", str(test_set))
    print("Recognized form: ", str(test_set_output))
    print(neural_network.table);

if __name__ == '__main__':
    main()
