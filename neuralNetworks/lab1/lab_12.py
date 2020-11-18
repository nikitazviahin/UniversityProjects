import numpy as np
from prettytable import PrettyTable



e = 0.0001
cols = ["Iteration", "Y", "Error"]


def save_weights(data):
    np.savetxt("wcoefs12.csv", data, delimiter=',')


class Perceptron:
    def __init__(this):

        #випадкові початкові вагові коеф.
        np.random.seed(1)
        # this.net_weights = 2 * np.random.random((2, 1)) - 1
        this.table = PrettyTable()
        this.net_weights = np.loadtxt("wcoefs12.csv", delimiter=',')

    @staticmethod
    #сігмоїдна функція для активації
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dfsigm(x):
        return x * (1 - x)

    def train(this, tr_in, tr_out, max_iter):
        #тренування мережі, tr_in матриця вхідних даних для тренування
        #tr_out матриця вихідних даних, max_iter якщо тренування буде невадлим для виходу з циклу
        iterations = []
        outputs = []
        errors = []

        y_third_layout = 0
        error = 0
        i = 0
        print("Start weights:", str(this.net_weights))
        for iteration in range(max_iter):
            #прохід даних через другий та третій слой
            x_second_layout = this.net_weights[0] * tr_in
            y_second_layout = this.sigm(x_second_layout)

            x_third_layout = this.net_weights[1] * y_second_layout
            y_third_layout = this.sigm(x_third_layout)

            error = abs((tr_out - y_third_layout) / tr_out)
            #виведення результатів частини ітерацій
            if (iteration + 1) % 50 == 0:
                iterations.append(iteration+1)
                outputs.append(round(y_third_layout[0][0], 6))
                errors.append(round(error[0][0], 6))

            if error <= e:
                print(this.net_weights)
                break
            #зворотній хід, метод розповсюдження помилок
            q_third_layout = this.dfsigm(y_third_layout) * (tr_out - y_third_layout)
            q_second_layout = this.dfsigm(y_second_layout) * (q_third_layout * this.net_weights[1])

            delta_weights_third_layout = q_third_layout * y_second_layout
            delta_weights_second_layout = q_second_layout * tr_in

            this.net_weights[0] += float(delta_weights_second_layout)
            this.net_weights[1] += float(delta_weights_third_layout)
            i += 1

        save_weights(this.net_weights)
        iterations.append(i + 1)
        outputs.append(round(y_third_layout[0][0], 6))
        errors.append(round(error[0][0], 6))

        this.table.add_column(cols[0], iterations)
        this.table.add_column(cols[1], outputs)
        this.table.add_column(cols[2], errors)

        print(this.table)

        print("Weights:", str(this.net_weights))

    def activate(this, inputs):
        #пропуск тренувального прикладу через нейронну мережу
        #повертаємо результат виконання
        x_second_layout = this.net_weights[0] * inputs
        y_second_layout = this.sigm(x_second_layout)

        x_third_layout = this.net_weights[1] * y_second_layout
        y_third_layout = this.sigm(x_third_layout)

        return y_third_layout


def main():
    tr_input = np.array([[5]])
    tr_output = np.array([0.2]).T
    perceptron = Perceptron()
    #тренування мережі
    perceptron.train(tr_input, tr_output, 1000)
    #режим розпізнавання
    test_input = np.array([[8]])
    result = perceptron.activate(test_input)
    print("Recognition regime: ")
    print("Initial vector:", str(tr_input))
    print("Recognized value: ", str(result))
    print("y = 0.2")


if __name__ == "__main__":
    main()
