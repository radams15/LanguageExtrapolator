import extrapolator

CFG = {
	"randomness": 0.1,
	"learn_times": 5,
	"gen_amount": 20,
}

if __name__ == '__main__':
	e = extrapolator.Extrapolator()
	
	data = open("training_data.txt", "r").read().replace("\n", "").split("****")

	e.finetune_list(data, cfg=CFG)
	
	e.generate_file(CFG["gen_amount"], "out_data.txt", randomness=CFG["randomness"])
