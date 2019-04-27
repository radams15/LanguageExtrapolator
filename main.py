import extrapolator

LEARN_TIMES = 20
NUMBER_TO_GENERATE = 5

if __name__ == '__main__':
    e = extrapolator.Extrapolator()

    e.load("biology.hdf5")

    e.finetune("training_data.txt", LEARN_TIMES)

    data = e.generate(NUMBER_TO_GENERATE)
    print("\n".join(data))

    e.save("biology.hdf5")