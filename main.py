import extrapolator

LEARN_TIMES = 1
NUMBER_TO_GENERATE = 50

if __name__ == '__main__':
    e = extrapolator.Extrapolator()

    e.finetune("training_data.txt", LEARN_TIMES)

    e.generate_file(NUMBER_TO_GENERATE, "trained_output.txt")