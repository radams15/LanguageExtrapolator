import extrapolator

CFG = {
	"randomness": 0.1,
	"learn_times": 1,
	"gen_amount": 1,
	"test_amount": 0 # generate while finetuning amount
}

if __name__ == '__main__':
	e = extrapolator.Extrapolator()
	
	data = open("training_data.txt", "r").read()\

	#data = data.replace("\n", "").split("****")

	e.finetune_list(data, cfg=CFG)
	
	e.generate_file(CFG["gen_amount"], "out_data.txt", randomness=CFG["randomness"], prefix="Trump Thinks")
