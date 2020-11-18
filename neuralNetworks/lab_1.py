from numpy import exp, array, random, dot, savetxt, loadtxt
from prettytable import PrettyTable
import numpy

EPSILON = 0.01
COLUMNS = ["Iteration", "Y", "Error"]


def save_weights(data):
    savetxt("weights_1_1.csv", data, delimiter=',')


class NeuralNetwork:
    def __init__(self):
        """
        Initialize start random weights
        dimension is 4x1
        """

        # For generation the same random weights every time
        random.seed(1)
        # self.synaptic_weights = 2*random.random((4, 1)) - 1
        self.synaptic_weights = loadtxt("weights_1_1.csv", delimiter=',')
        self.table = PrettyTable()

    @staticmethod
    def __sigmoid(x):
        """
        Calculate sigmoid function
        """
        return 1 / (1 + exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        """
        Calculate derivative of sigmoid function
        """
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations):
        """
        Trainings iterations

        :param training_set_inputs: matrix of training set
        :param training_set_outputs: matrix of results of training set
        :param max_number_of_training_iterations: int, for breaking the loop if error > epsilon
        :return:
        """
        iterations = []
        outputs = []
        errors = []

        output = 0
        error = 0
        i = 0

        print("Start weights:", str(self.synaptic_weights))
        for iteration in range(max_number_of_training_iterations):
            # print("-------------------------------------------------")
            # print("Iteration:\t", iteration+1)

            output = self.activate(training_set_inputs)

            error = abs((training_set_outputs - output) / training_set_outputs)
            # print("Error:\t", error[0][0])

            if (iteration + 1) % 10 == 0:
                iterations.append(iteration+1)
                outputs.append(numpy.round(output[0], 6))
                errors.append(numpy.round(error[0], 6))

            # with open("lab_1_1_temp.txt", "a") as f:
            #     f.write(str(iteration)+'\t'+str(error[0][0])+'\n')

            if error < EPSILON:
                break

            delta = self.__sigmoid_derivative(output) * (training_set_outputs - output)
            adjustment = dot(training_set_inputs.T, delta)
            for j in range(len(self.synaptic_weights)):
                self.synaptic_weights[j] += adjustment[j][0]
            # print("Synaptic weights:\t", self.synaptic_weights)
            # print("-------------------------------------------------")
            i += 1
        save_weights(self.synaptic_weights)
        iterations.append(i + 1)
        outputs.append(numpy.round(output[0], 6))
        errors.append(numpy.round(error[0], 6))

        self.table.add_column(COLUMNS[0], iterations)
        self.table.add_column(COLUMNS[1], outputs)
        self.table.add_column(COLUMNS[2], errors)

        print(self.table)

        print("Weights:", str(self.synaptic_weights))

    def activate(self, inputs):
        """
        Pass inputs through our neural network

        :param inputs: training example
        :return: result of neural network
        """

        return self.__sigmoid(dot(inputs, self.synaptic_weights))


def save_result(input_data, output_data):
    with open("lab_1_1_result.txt", 'w') as f:
        f.write(str(input_data)+'\t'+str(output_data))


def main():
    #   Initialize a neural network
    neural_network = NeuralNetwork()

    #   Define training set
    training_set_inputs = array([[1, 7, 4, 5]])
    training_set_outputs = array([[0.3]]).T

    #   Start training
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    test_set = array([1, 7, 4, 5])
    test_set_output = neural_network.activate(test_set)
    print("Режим розпізнавання: ")
    print("Початковий вектор:", str(test_set))
    print("Розпізнаний образ: ", str(test_set_output))


if __name__ == '__main__':
    main()
