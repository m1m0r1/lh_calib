_calib_mixed:
	$(eval ds = mixed_mnist)
	$(eval model = vgg16)
	$(eval setting = )
	$(eval epochs = 50)
	$(eval early_stop = 10)
	$(eval batch_size = 128)
	$(eval input_tab = $(LH_BASE)/data/$(ds)/$(in_name).txt.gz)
	$(eval run_prefix = $(LH_BASE)/results/$(ds)/$(run_name))
	$(eval model_path = $(run_prefix).model.h5)
	$(eval model_weights = $(run_prefix).best.weights.h5)
	$(eval calib_prefix = $(run_prefix).best$(calib_ds_suf).$(calib_name))
	$(eval main_opts = -i $(input_tab) -m $(model) -s $(setting) -mp $(model_path) -mw $(model_weights) -o $(calib_prefix) -b $(batch_size) -o $(calib_prefix) -e $(epochs) -es $(early_stop))
	$(PYTHON) $(LH_BASE)/scripts/calibrate.py -v $(main_opts) $(opts)
	$(eval partition = test)
	$(eval calib_model_path = $(calib_prefix).model.h5)
	$(eval pred_prefix = $(calib_prefix).best)
	$(eval calib_model_weights = $(pred_prefix).weights.h5)
	$(eval pred_opts = -i $(input_tab) -m $(model) -s $(setting) -mp $(calib_model_path) -mw $(calib_model_weights) -b $(batch_size) -p $(partition) -st $(score_type))
	$(eval mc_samples = 20)
	$(eval pred_opts1 = ) $(eval pred_suf = )
	$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz)
	$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output))
	$(eval pred_opts1 = --use-dropout -mcs $(mc_samples)) $(eval pred_suf = .pmcdo.20)
	$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz)
	$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output))
	$(if $(filter-out mixed_mnist, $(ds)), \
		$(eval pred_opts1 = --use-augment -mcs $(mc_samples)) $(eval pred_suf = .tta.20) \
		$(eval output = $(pred_prefix).$(partition).scores$(pred_suf).txt.gz) \
		$(PYTHON) $(LH_BASE)/scripts/predict.py -v $(pred_opts) $(pred_opts1) -o >(gzip -c > $(output)); \
	)


# first phase calibration
calib_mixed_mnist_p1 \
calib_mixed_cifar10_p1 \
: calib_%_p1 :
	$(eval train_ds_suf = .t1_v1)   # used for training
	$(eval calib_ds_suf = .t1_v2)   # empty means all labels
	$(eval ds = $*)
	$(eval setting = $(ds))
	$(if $(filter mixed_mnist, $(ds)), $(eval model = simple_cnn) $(eval model_suf = .cnn),)
	$(if $(filter mixed_cifar10, $(ds)), $(eval model = vgg16) $(eval model_suf = .vgg16),)
	$(eval in_name = $(ds))
	$(eval calib_in_name = $(ds)$(calib_ds_suf))
	# scaling
	$(eval job_prefix = $(in_name)$(calib_ds_suf)$(model_suf))
	$(eval run_name = $(in_name)$(train_ds_suf)$(model_suf)/run)
	$(eval score_type = prob)
	$(eval opts = -lc ts ) $(eval calib_name = ts) $(eval jobname = $(job_prefix).$(calib_name))
	$(_RUN_GPU) make _calib_mixed ds=$(ds) calib_ds_suf=$(calib_ds_suf)  in_name=$(calib_in_name) model=$(model) setting=$(setting) run_name=$(run_name) calib_name=$(calib_name) opts="$(opts)" score_type=$(score_type)
	$(eval score_type = alpha)
	$(eval opts = -la -fl -2 --log-alpha0-l2 0.005 -a0ar) $(eval calib_name = la_la0_0005_ar) $(eval jobname = $(job_prefix).$(calib_name))
	$(_RUN_GPU) make _calib_mixed ds=$(ds) calib_ds_suf=$(calib_ds_suf) in_name=$(calib_in_name) model=$(model) setting=$(setting) run_name=$(run_name) calib_name=$(calib_name) opts="$(opts)" score_type=$(score_type)


# second phase calibration
calib_mixed_mnist_p2 \
calib_mixed_cifar10_p2 \
: calib_%_p2 :
	$(eval train_ds_suf = .t1_v1)   # used for training
	$(eval calib_ds_suf = .t1_v2)
	$(eval ds = $*)
	$(eval setting = $(ds))
	$(if $(filter mixed_mnist, $(ds)), $(eval model = simple_cnn) $(eval model_suf = .cnn),)
	$(if $(filter mixed_cifar10, $(ds)), $(eval model = vgg16) $(eval model_suf = .vgg16),)
	$(eval in_name = $(ds))
	$(eval calib_in_name = $(ds)$(calib_ds_suf))
	# scaling
	$(eval run_name = $(in_name)$(train_ds_suf)$(model_suf)/run)
	$(eval score_type = alpha)
	# post-hoc alpha calibration jobs after logit calibration
	$(eval opts = -la -fl -3 --log-alpha0-l2 0.005 -a0ar) $(eval calib_name = la_la0_0005_ar)
	$(eval lc_suffix = .best$(calib_ds_suf).ts) $(eval jobname = $(in_name)$(calib_ds_suf)$(model_suf)$(lc_suffix).$(calib_name))
	$(eval job_opts = -hold_jid $(in_name)$(calib_ds_suf)$(model_suf).ts)
	$(_RUN_GPU) make _calib_mixed ds=$(ds) calib_ds_suf=$(calib_ds_suf) in_name=$(calib_in_name) model=$(model) setting=$(setting) run_name=$(run_name)$(lc_suffix) calib_name=$(calib_name) opts="$(opts)" score_type=$(score_type)


calib_mixed_mnist_p1s \
calib_mixed_cifar10_p1s \
calib_mixed_mnist_p2s \
calib_mixed_cifar10_p2s \
: calib_%s : \
	$(eval train_ds_suf = .t1_v1) $(eval calib_ds_suf = .t1_v2)
	make calib_$* train_ds_suf=$(train_ds_suf) calib_ds_suf=$(calib_ds_suf)
	$(eval train_ds_suf = .t1_v1) $(eval calib_ds_suf = .t1_v5)
	make calib_$* train_ds_suf=$(train_ds_suf) calib_ds_suf=$(calib_ds_suf)

calib_mixed_p1s:
	make calib_mixed_mnist_p1s
	make calib_mixed_cifar10_p1s

calib_mixed_p2s:
	make calib_mixed_mnist_p2s
	make calib_mixed_cifar10_p2s
