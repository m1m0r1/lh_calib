_pred_post_mixed_mnist \
_pred_post_mixed_cifar10 \
: _pred_post_% :
	$(eval ds = $*)
	$(eval train_ds_suf = .t1_v1)
	$(eval in_name = $(ds))   # test set is common for all the dataset
	$(if $(filter mixed_mnist, $(ds)), $(eval model_suf = .cnn),)
	$(if $(filter mixed_cifar10, $(ds)), $(eval model_suf = .vgg16),)
	$(eval sver = n1_s1)
	$(eval partition = test)
	$(eval pred_suf = )   # e.g., .pmcdo.20 and .tta.20
	$(eval input_tab = $(LH_BASE)/data/$(ds)/$(in_name).test_split.$(sver).sam.txt.gz)
	$(eval out_name = $(ds)$(train_ds_suf)$(model_suf))
	$(eval run_prefix = $(LH_BASE)/results/$(ds)/$(out_name)/run)
	$(eval calib_suf = )
	$(eval prefix = $(run_prefix)$(calib_suf).best.$(partition).scores$(pred_suf))
	$(eval prior = $(prefix).txt.gz)
	$(eval posterior = $(prefix).post.$(sver).txt.gz)
	$(PYTHON) $(LH_BASE)/scripts/predict_posterior.py -v -i $(input_tab) -p $(partition) -pr $(prior) $(opts) -o >(gzip -c > $(posterior))

pred_post_mixed_mnist \
pred_post_mixed_cifar10 \
: pred_post_% :
	$(eval ds = $*)
	$(eval in_name = $(ds))
	$(eval calib_ds_suf = .t1_v2)
	$(eval calib_suf = )
	$(eval jobname = posterior)
	$(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf)  pred_suf=.pmcdo.20
	$(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf)  pred_suf=.pmcdo.20
	$(if $(filter-out mixed_mnist, $(ds)), $(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf) pred_suf=.tta.20;)
	$(eval calib_suf = .best$(calib_ds_suf).ts)
	$(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf)  pred_suf=.pmcdo.20
	$(if $(filter-out mixed_mnist, $(ds)), $(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf) pred_suf=.tta.20;)
	$(eval calib_name = la_la0_0005_ar)
	$(eval calib_suf = .best$(calib_ds_suf).$(calib_name))
	$(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf)
	$(eval calib_suf = .best$(calib_ds_suf).ts.best$(calib_ds_suf).$(calib_name))
	$(_RUN_CPU) make _pred_post_$(ds) in_name=$(in_name) calib_suf=$(calib_suf)

pred_post_mixed:
	make pred_post_mixed_mnist   calib_ds_suf=.t1_v2
	make pred_post_mixed_mnist   calib_ds_suf=.t1_v5
	make pred_post_mixed_cifar10 calib_ds_suf=.t1_v2
	make pred_post_mixed_cifar10 calib_ds_suf=.t1_v5
