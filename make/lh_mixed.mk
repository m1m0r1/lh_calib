_train_mixed:
	$(eval input_tab = $(LH_BASE)/data/$(ds)/$(in_name).txt.gz)
	$(eval output = $(LH_BASE)/results/$(ds)/$(out_name))
	$(eval opts = -b 128)
	python $(LH_BASE)/scripts/train.py train_lh -i $(input_tab) -o $(output) -v $(opts) --save-weight last --save-period 0 --save-best

train_mixed_mnist:
	$(eval epochs = 100)
	$(eval batch_size = 128)
	$(eval ds = mixed_mnist)
	$(eval model = simple_cnn) $(eval model_loss = multinom_nll) $(eval setting = mixed_mnist)
	$(eval opts = -b $(batch_size) -e $(epochs) -ml $(model_loss) -m $(model) -s $(setting) -es 10)
	$(eval in_name = $(ds).t1_v1)
	$(eval out_name = $(in_name).cnn) $(eval jobname = $(out_name))
	$(_RUN_GPU) make _train_mixed ds=$(ds) in_name=$(in_name) out_name=$(out_name) opts="$(opts)"

train_mixed_cifar10:
	$(eval epochs = 250)
	$(eval batch_size = 128)
	$(eval ds = mixed_cifar10)
	$(eval model = vgg16) $(eval model_loss = multinom_nll) $(eval setting = mixed_cifar10)
	$(eval opts = -b $(batch_size) -e $(epochs) -ml $(model_loss) -m $(model) -s $(setting) --use-warmup --ms-decay-epochs 100 150)
	$(eval in_name = $(ds).t1_v1)
	$(eval out_name = $(in_name).vgg16) $(eval jobname = $(out_name))
	$(_RUN_GPU) make _train_mixed ds=$(ds) in_name=$(in_name) out_name=$(out_name) opts="$(opts)"


_pred_mixed:
	$(eval ds = mixed_mnist)
	$(eval batch_size = 128)
	$(eval model = vgg16)
	$(eval setting = $(ds))
	$(eval partition = test)
	$(eval input_tab = $(LH_BASE)/data/$(ds)/$(in_name).txt.gz)
	$(eval run_prefix = $(LH_BASE)/results/$(ds)/$(out_name)/run)
	$(eval model_path = $(run_prefix).model.h5)
	$(eval model_weights = $(run_prefix).best.weights.h5)
	$(eval pred_prefix = $(run_prefix).best)
	$(eval pred_opts = -i $(input_tab) -m $(model) -s $(setting) -mp $(model_path) -mw $(model_weights) -b $(batch_size) -p $(partition) -st $(score_type))
	$(eval pred_opts1 = ) $(eval pred_suf = )
	$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz)
	$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output))
	$(eval pred_opts1 = --use-dropout -mcs 20) $(eval pred_suf = .pmcdo.20)
	$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz)
	$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output))
	$(if $(filter-out mixed_mnist, $(ds)), \
		$(eval pred_opts1 = --use-augment -mcs 20) $(eval pred_suf = .tta.20) \
		$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz) \
		$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output)); \
	)

pred_mixed_mnist \
pred_mixed_cifar10 \
: pred_% :
	$(eval ds = $*)
	$(if $(filter mixed_mnist, $(ds)), $(eval model = simple_cnn) $(eval model_suf = .cnn),)
	$(if $(filter mixed_cifar10, $(ds)), $(eval model = vgg16) $(eval model_suf = .vgg16),)
	$(eval score_type = prob)
	$(eval train_suf = .t1_v1)
	$(eval in_name = $(ds)$(train_suf))
	$(eval out_name = $(ds)$(train_suf)$(model_suf))
	$(eval jobname = $(out_name))
	$(_RUN_GPU) make _pred_mixed ds=$(ds) model=$(model) in_name=$(in_name) out_name=$(out_name) score_type=$(score_type)
