from textgenrnn import textgenrnn

class Extrapolator:
	def __init__(self):
		self.gen = textgenrnn()

	def finetune_file(self, file, cfg, steps=1, gen=0):
		self.gen.train_from_file(
				file, 
				num_epochs=cfg["learn_times"],
				gen_epochs=cfg["gen_amount"]
			)
		
	def finetune_list(self, list_of_text, cfg, steps=1, gen=0):
		self.gen.train_on_texts(list_of_text,
				num_epochs=cfg["learn_times"],
				gen_epochs=cfg["gen_amount"]
			)

	def save(self, out_file):
		self.gen.save(out_file)

	def load(self, in_file):
		self.gen = textgenrnn(in_file)

	def generate(self, amount, randomness=1):
		return self.gen.generate(amount, temperature=randomness, return_as_list=True)

	def generate_file(self, amount, out_file, randomness=1):
		self.gen.generate_to_file(out_file, n=amount, temperature=randomness)
