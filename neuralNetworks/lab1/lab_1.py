from numpy import exp, array, random, dot, savetxt, loadtxt
from prettytable import PrettyTable
import numpy

e = 0.001
cols = ["Iteration", "Y", "Error"]


def save_weights(data):
    savetxt("wcoefs11.csv", data, delimiter=',')


class NeuralNetwork:
    def __init__(this):

        #Випадкові початкові вагові коеф-ти
        random.seed(1)
        # this.net_weights = 2*random.random((4, 1)) - 1
        this.net_weights = loadtxt("wcoefs11.csv", delimiter=',')
        this.table = PrettyTable()


    #сигмоїда як функція активації
    @staticmethod
    def sigm(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def dfsigm(x):
        return x * (1 - x)

    def train(this, tr_in, tr_out, max_iter):
        #тренування мережі, tr_in матриця вхідних даних для тренування
        #tr_out матриця вихідних даних, max_iter якщо тренування буде невадлим для виходу з циклу
        iterations = []
        outputs = []
        errors = []

        output = 0
        error = 0
        i = 0

        print("Start weights:", str(this.net_weights))
        for iteration in range(max_iter):

            #обрахування похибки після проходу вектору через мережу
            output = this.activate(tr_in)
            error = abs((tr_out - output) / tr_out)

            #виведення частини ітерацій
            if (iteration + 1) % 50 == 0:
                iterations.append(iteration+1)
                outputs.append(numpy.round(output[0], 6))
                errors.append(numpy.round(error[0], 6))

            if error < e:
                break

            delta = this.dfsigm(output) * (tr_out - output)
            adjustment = dot(tr_in.T, delta)
            for j in range(len(this.net_weights)):
                this.net_weights[j] += adjustment[j][0]

            i += 1
        save_weights(this.net_weights)
        iterations.append(i + 1)
        outputs.append(numpy.round(output[0], 6))
        errors.append(numpy.round(error[0], 6))

        this.table.add_column(cols[0], iterations)
        this.table.add_column(cols[1], outputs)
        this.table.add_column(cols[2], errors)

        print(this.table)

        print("Weights:", str(this.net_weights))

    def activate(this, inputs):
        #пропуск тренувального прикладу через нейронну мережу
        #повертаємо результат виконання
        return this.sigm(dot(inputs, this.net_weights))


def save_result(input_data, output_data):
    with open("lab_1_1_result.txt", 'w') as f:
        f.write(str(input_data)+'\t'+str(output_data))


def main():
    #   Initialize a neural network
    neural_network = NeuralNetwork()

    #   Define training set
    tr_in = array([[1, 7, 4, 5]])
    tr_out = array([[0.5]]).T

    #   Start training
    neural_network.train(tr_in, tr_out, 1000)

    test_set = array([1, 7, 4, 5])
    test_set_output = neural_network.activate(test_set)
    print("Recognition regime: ")
    print("Initial vector:", str(test_set))
    print("Recognized value: ", str(test_set_output))


if __name__ == '__main__':
    main()
