debias_sim:
	mkdir -p $(LH_BASE)/sim
	python $(LH_BASE)/lh_calib/simulation/debias_effect.py gen_and_eval_basic \
		-v \
		-N 100 250 500 1000 2500 5000 10000 \
		-n 2 5 \
		-r 10 > $(LH_BASE)/sim/debias_effect1.csv
