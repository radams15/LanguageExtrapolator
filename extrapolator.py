from textgenrnn import textgenrnn

class Extrapolator:
    def __init__(self):
        self.gen = textgenrnn()

    def finetune(self, file, steps=1):
        self.gen.train_from_file(file, num_epochs=steps)

    def save(self, out_file):
        self.gen.save(out_file)

    def load(self, in_file):
        self.gen.load(in_file)

    def generate(self, amount, temperature=1):
        return self.gen.generate(amount, temperature=temperature, return_as_list=True)