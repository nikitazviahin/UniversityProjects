import numpy as np
from prettytable import PrettyTable

#дані для тренування
train_in = np.array([[3, 2]])
train_out = np.array([[0.5]])

#дані для тестування
recognize_input = np.array([[4, 1.5]])

cols = ["Iteration", "Y", "Error"]
e = 0.01
features_num = len(train_in)


class Perceptron:
    def __init__(this):
        

        this.table = PrettyTable()
        #ініціалізуємо початкові ваги в окремих файлах, розмірність 2х3 та 3х1
        this.hidden_w = np.loadtxt('hidden_weights_online.csv', delimiter=',')
        this.out_w = np.loadtxt('output_weights_online.csv', delimiter=',')

    @staticmethod
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dfsigm(x):
        return x * (1 - x)

    def train(this, tr_inputs, tr_outs, max_iter):
        
        #тренування мережі, tr_in матриця вхідних даних для тренування
        #tr_out матриця вихідних даних, max_iter якщо тренування буде невадлим для виходу з циклу

        iterations = []
        outputs = []
        errors = []

        output_op = 0
        error = 0
        i = 0

        print("Початкові скриті ваги:", str(this.hidden_w))
        print("Початкові ваги на виході:", str(this.out_w))
        for iteration in range(max_iter):
            input_hidden = np.dot(tr_inputs, this.hidden_w)

            #Виведення зі скритого слою
            output_hidden = this.sigm(input_hidden)

            #Введення для вихідного слою
            input_op = np.dot(output_hidden, this.out_w)

            #Виведення для вихідного слою
            output_op = this.sigm(input_op)

            #Підрахування похибки та виведення частини даних у консоль
            error = abs((tr_outs - output_op) / tr_outs)

            if (iteration + 1) % 10 == 0:
                iterations.append(iteration+1)
                outputs.append(round(output_op[0], 6))
                errors.append(round(error[0][0], 6))

            if error <= e:
                break

            #зворотнє розповсюдження помилки
            q_out_lay = this.dfsigm(output_op) * (tr_outs - output_op)
            del_out_lay = q_out_lay * output_hidden

            del_out_lay = np.reshape(del_out_lay, (3, 1))
            q_hid_1 = this.dfsigm(
                output_hidden[0][0]) * (q_out_lay * this.out_w[0])
            del_hid_1 = q_hid_1 * tr_inputs

            q_hidden_layout_2 = this.dfsigm(
                output_hidden[0][1]) * (q_out_lay * this.out_w[1])
            del_hid_2 = q_hidden_layout_2 * tr_inputs

            q_hidden_layout_3 = this.dfsigm(
                output_hidden[0][2]) * (q_out_lay * this.out_w[2])
            del_hid_3 = q_hidden_layout_3 * tr_inputs

            delta_hidden_layout = np.array(
                [[del_hid_1[0][0], del_hid_2[0][0], del_hid_3[0][0]],
                 [del_hid_1[0][1], del_hid_2[0][1], del_hid_3[0][1]]])

            for i in range(3):
                this.out_w[i] += del_out_lay[i][0]
            this.hidden_w += delta_hidden_layout

            i += 1

        
        #Збереження результатів
        save_result(this.hidden_w, this.out_w)

        iterations.append(i + 1)
        outputs.append(round(output_op[0], 6))
        errors.append(round(error[0][0], 6))

        this.table.add_column(cols[0], iterations)
        this.table.add_column(cols[1], outputs)
        this.table.add_column(cols[2], errors)

        print(this.table)

        print("Скриті вагові коеф-ти:\n ", str(this.hidden_w))
        print("Вихідні вагові коеф-ти:\n ", str(this.out_w))

    def online_train(this, tr_inputs, tr_outs, epochs):
        outputs = []
        errors = []
        iterations = []

        print("Початкові скрити вагові коеф-ти:\n", str(this.hidden_w))
        print("Початкові вихідні вагові коеф-ти\n", str(this.out_w))
        for epoch in range(epochs):
            input_hidden = np.dot(tr_inputs, this.hidden_w)

            output_hidden = this.sigm(input_hidden)

            input_op = np.dot(output_hidden, this.out_w)

            output_op = this.sigm(input_op)

            outputs.append(output_op[0])
            #error = abs((tr_outs - np.reshape(output_op, (4, 1))) / np.reshape(output_op, (4, 1)))
            error = abs((tr_outs - output_op) / output_op )
            print('Епоха: ', epoch + 1)
            print('Похибка: ', error)
            print('Розпізнаний образ:', outputs[epoch])
            if error <= e:
                break

            errors.append(error[0])
            iterations.append(epoch + 1)

            for i in range(features_num):
                q_out_lay = this.dfsigm(output_op[i]) * (tr_outs[i] - output_op[i])
                del_out_lay = q_out_lay * output_hidden[i]
                del_out_lay = np.reshape(del_out_lay, (3, 1))

                q_hid_1 = this.dfsigm(
                    output_hidden[i][0]) * (q_out_lay * this.out_w[0])
                del_hid_1 = q_hid_1 * tr_inputs[i]

                q_hidden_layout_2 = this.dfsigm(
                    output_hidden[i][1]) * (q_out_lay * this.out_w[1])
                del_hid_2 = q_hidden_layout_2 * tr_inputs[i]

                q_hidden_layout_3 = this.dfsigm(
                    output_hidden[i][2]) * (q_out_lay * this.out_w[2])
                del_hid_3 = q_hidden_layout_3 * tr_inputs[i]

                delta_hidden_layout = np.array(
                    [[del_hid_1[0], del_hid_2[0], del_hid_3[0]],
                     [del_hid_1[1], del_hid_2[1], del_hid_3[1]]])


                print(this.out_w)
                print(del_out_lay)
                for i in range(3):
                    this.out_w[i] += del_out_lay[i][0]
                this.hidden_w += delta_hidden_layout


        save_result(this.hidden_w, this.out_w)

        print("Скриті вагові коеф-ти:", str(this.hidden_w))
        print("Вагові коеф-ти на виході:", str(this.out_w))

    def activate(this, inputs):
       
        #подача даних та їх прохід через нейронну мережу
        #Виведення зі скритого слою
        input_hidden = np.dot(inputs, this.hidden_w)

        #Введення для вихідного слою
        output_hidden = this.sigm(input_hidden)

        #Виведення для вихідного слою
        input_op = np.dot(output_hidden, this.out_w)

        output_op = this.sigm(input_op)

        return output_op


def save_result(data_1, data_2):
    np.savetxt("hidden_weights_online.csv", data_1, delimiter=',')
    np.savetxt("output_weights_online.csv", data_2, delimiter=',')


def main():
    perceptron = Perceptron()

    #perceptron.train(train_in, train_out, 10000)
    perceptron.online_train(train_in, train_out, 1000000)

    result = perceptron.activate(recognize_input)
    print("Режим розпізнавання: ")
    print("Початковий вектор:", str(recognize_input))
    print("Розпізнаний образ: ", str(result))


if __name__ == "__main__":
    main()



