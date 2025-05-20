(.venv_param_recovery) (base) PS C:\Users\jesse\Hegemonikon Project\hegemonikon> python run_parameter_recovery_minimal_nes_npe.py --n_subj 30 --n_trials 100 --npe_train_sims 20000 --npe_posterior_samples 500
WARNING (pytensor.configdefaults): g++ not available, if using conda: `conda install gxx`
WARNING (pytensor.configdefaults): g++ not detected!  PyTensor will be unable to compile C-implementations and will default to Python. Performance may be severely degraded. To remove this warning, set PyTensor flags cxx to an empty string.
Successfully imported sbi version: 0.24.0
Successfully imported arviz version: 0.21.0
Using device: cpu
============================================================
Starting Parameter Recovery for Minimal NES Parameters using NPE
============================================================

--- Training NPE for Minimal NES Recovery ---
Using 20000 simulations for training.
Using 20000 valid simulations for training NPE.
 Neural network successfully converged after 138 epochs.NPE training took: 1284.22s

--- Running Recovery for 30 Synthetic Subjects ---

------------------------------
Processing Subject 1/30
  True parameters for subject 1: {'v_norm': 1.6152827739715576, 'a_0': 1.1161861419677734, 'w_s_eff': 0.27978822588920593, 't_0': 0.09210287779569626}
Observed summary stats for subject 1:
  choice_rate_overall: 0.22
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.6818181818181818
  error_rate_lvl_0_50: 0.9473684210526315
  error_rate_lvl_0_75: 0.8125
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 8.382862633115048
  overall_rt_min: 0.2721028777956963
  rt_mean_correct_lvl_0_00: 2.4057392414320407
  rt_mean_correct_lvl_0_25: 4.0335314492242125
  rt_mean_correct_lvl_0_50: 0.4121028777956964
  rt_mean_correct_lvl_0_75: 1.0587695444623606
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 1 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 2/30
  True parameters for subject 2: {'v_norm': 0.49444642663002014, 'a_0': 1.6624809503555298, 'w_s_eff': 0.6608006358146667, 't_0': 0.2813142240047455}
Observed summary stats for subject 2:
  choice_rate_overall: 0.63
  error_rate_lvl_0_00: 0.05263157894736842
  error_rate_lvl_0_25: 0.09090909090909091
  error_rate_lvl_0_50: 0.2777777777777778
  error_rate_lvl_0_75: 0.6363636363636364
  error_rate_lvl_1_00: 0.7894736842105263
  overall_rt_mean: 6.021427961122968
  overall_rt_min: 0.5713142240047455
  rt_mean_correct_lvl_0_00: 3.1057586684491656
  rt_mean_correct_lvl_0_25: 3.6998142240047107
  rt_mean_correct_lvl_0_50: 4.901314224004685
  rt_mean_correct_lvl_0_75: 3.6850642240047105
  rt_mean_correct_lvl_1_00: 2.2613142240047317
============================================================
Drawing 500 posterior samples: 549it [00:00, 17076.06it/s]

------------------------------
Processing Subject 3/30
  True parameters for subject 3: {'v_norm': 0.45165514945983887, 'a_0': 1.6484211683273315, 'w_s_eff': 0.7406688928604126, 't_0': 0.14973586797714233}
Observed summary stats for subject 3:
  choice_rate_overall: 0.61
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.42105263157894735
  error_rate_lvl_0_75: 0.5454545454545454
  error_rate_lvl_1_00: 0.76
  overall_rt_mean: 5.303238879466048
  overall_rt_min: 0.46973586797714245
  rt_mean_correct_lvl_0_00: 1.7325930108342809
  rt_mean_correct_lvl_0_25: 2.3182358679771293
  rt_mean_correct_lvl_0_50: 3.1879176861589325
  rt_mean_correct_lvl_0_75: 2.3627358679771304
  rt_mean_correct_lvl_1_00: 1.8347358679771377
============================================================
Drawing 500 posterior samples: 599it [00:00, 25749.07it/s]

------------------------------
Processing Subject 4/30
  True parameters for subject 4: {'v_norm': 1.199280023574829, 'a_0': 0.7956901788711548, 'w_s_eff': 1.2328637838363647, 't_0': 0.11066649854183197}
Observed summary stats for subject 4:
  choice_rate_overall: 0.68
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.1935483870967742
  error_rate_lvl_0_75: 0.6
  error_rate_lvl_1_00: 0.7777777777777778
  overall_rt_mean: 4.086253219008442
  overall_rt_min: 0.17066649854183197
  rt_mean_correct_lvl_0_00: 0.7163807842561172
  rt_mean_correct_lvl_0_25: 1.3165488514830046
  rt_mean_correct_lvl_0_50: 1.7830664985418203
  rt_mean_correct_lvl_0_75: 1.0706664985418304
  rt_mean_correct_lvl_1_00: 0.7681664985418324
============================================================
Drawing 500 posterior samples: 568it [00:00, 17788.66it/s]

------------------------------
Processing Subject 5/30
  True parameters for subject 5: {'v_norm': 0.7111079096794128, 'a_0': 1.7448722124099731, 'w_s_eff': 1.2001756429672241, 't_0': 0.25331929326057434}
Observed summary stats for subject 5:
  choice_rate_overall: 0.65
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.14814814814814814
  error_rate_lvl_0_75: 0.7307692307692307
  error_rate_lvl_1_00: 0.9230769230769231
  overall_rt_mean: 5.166157540619363
  overall_rt_min: 0.48331929326057443
  rt_mean_correct_lvl_0_00: 1.6008192932605696
  rt_mean_correct_lvl_0_25: 2.098874848816122
  rt_mean_correct_lvl_0_50: 3.544188858477934
  rt_mean_correct_lvl_0_75: 2.8733192932605545
  rt_mean_correct_lvl_1_00: 1.5933192932605753
============================================================
Drawing 500 posterior samples: 539it [00:00, 17554.29it/s]

------------------------------
Processing Subject 6/30
  True parameters for subject 6: {'v_norm': 1.3221853971481323, 'a_0': 1.2765586376190186, 'w_s_eff': 1.0159164667129517, 't_0': 0.341187447309494}
Observed summary stats for subject 6:
  choice_rate_overall: 0.49
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.06666666666666667
  error_rate_lvl_0_50: 0.5909090909090909
  error_rate_lvl_0_75: 0.9583333333333334
  error_rate_lvl_1_00: 0.9333333333333333
  overall_rt_mean: 6.2460818491816426
  overall_rt_min: 0.5511874473094941
  rt_mean_correct_lvl_0_00: 2.218270780642815
  rt_mean_correct_lvl_0_25: 1.852616018738056
  rt_mean_correct_lvl_0_50: 3.6845207806427913
  rt_mean_correct_lvl_0_75: 1.5511874473094949
  rt_mean_correct_lvl_1_00: 0.7211874473094941
============================================================
Drawing 500 posterior samples: 583it [00:00, 20438.30it/s]

------------------------------
Processing Subject 7/30
  True parameters for subject 7: {'v_norm': 1.1397653818130493, 'a_0': 1.592677354812622, 'w_s_eff': 0.5102653503417969, 't_0': 0.1593208611011505}
Observed summary stats for subject 7:
  choice_rate_overall: 0.38
  error_rate_lvl_0_00: 0.0625
  error_rate_lvl_0_25: 0.42857142857142855
  error_rate_lvl_0_50: 0.6363636363636364
  error_rate_lvl_0_75: 0.8571428571428571
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.4289419272184265
  overall_rt_min: 0.6393208611011507
  rt_mean_correct_lvl_0_00: 3.263987527767787
  rt_mean_correct_lvl_0_25: 3.3459875277677873
  rt_mean_correct_lvl_0_50: 3.620570861101117
  rt_mean_correct_lvl_0_75: 1.6059875277678184
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 7 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 8/30
  True parameters for subject 8: {'v_norm': 0.2755480110645294, 'a_0': 0.8453562259674072, 'w_s_eff': 0.7645630836486816, 't_0': 0.05642871931195259}
Observed summary stats for subject 8:
  choice_rate_overall: 0.92
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.045454545454545456
  error_rate_lvl_0_50: 0.0
  error_rate_lvl_0_75: 0.19047619047619047
  error_rate_lvl_1_00: 0.1875
  overall_rt_mean: 2.3032144217669868
  overall_rt_min: 0.2364287193119526
  rt_mean_correct_lvl_0_00: 1.2164287193119487
  rt_mean_correct_lvl_0_25: 2.4869049097881195
  rt_mean_correct_lvl_0_50: 1.6149287193119435
  rt_mean_correct_lvl_0_75: 1.1940757781354803
  rt_mean_correct_lvl_1_00: 1.53489025777348
============================================================
Drawing 500 posterior samples: 566it [00:00, 15989.71it/s]

------------------------------
Processing Subject 9/30
  True parameters for subject 9: {'v_norm': 0.7757723927497864, 'a_0': 1.9497621059417725, 'w_s_eff': 0.5603426694869995, 't_0': 0.12286078929901123}
Observed summary stats for subject 9:
  choice_rate_overall: 0.49
  error_rate_lvl_0_00: 0.05263157894736842
  error_rate_lvl_0_25: 0.047619047619047616
  error_rate_lvl_0_50: 0.5714285714285714
  error_rate_lvl_0_75: 0.875
  error_rate_lvl_1_00: 0.9090909090909091
  overall_rt_mean: 7.0096017867564955
  overall_rt_min: 0.5728607892990114
  rt_mean_correct_lvl_0_00: 3.405638567076755
  rt_mean_correct_lvl_0_25: 4.829360789298952
  rt_mean_correct_lvl_0_50: 2.7178607892989954
  rt_mean_correct_lvl_0_75: 4.319527455965633
  rt_mean_correct_lvl_1_00: 1.9028607892990093
============================================================
Drawing 500 posterior samples: 527it [00:00, 16073.40it/s]

------------------------------
Processing Subject 10/30
  True parameters for subject 10: {'v_norm': 1.9828007221221924, 'a_0': 0.834700882434845, 'w_s_eff': 1.1247280836105347, 't_0': 0.37384679913520813}
Observed summary stats for subject 10:
  choice_rate_overall: 0.5
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.1111111111111111
  error_rate_lvl_0_50: 0.5625
  error_rate_lvl_0_75: 0.9473684210526315
  error_rate_lvl_1_00: 0.9523809523809523
  overall_rt_mean: 5.788923399567601
  overall_rt_min: 0.4838467991352081
  rt_mean_correct_lvl_0_00: 1.0591409167822672
  rt_mean_correct_lvl_0_25: 1.9530134658018643
  rt_mean_correct_lvl_0_50: 1.7352753705637771
  rt_mean_correct_lvl_0_75: 1.3038467991352087
  rt_mean_correct_lvl_1_00: 0.5638467991352082
============================================================
Drawing 500 posterior samples: 559it [00:00, 16446.17it/s]

------------------------------
Processing Subject 11/30
  True parameters for subject 11: {'v_norm': 0.31081801652908325, 'a_0': 1.1501777172088623, 'w_s_eff': 1.1751993894577026, 't_0': 0.41360902786254883}
Observed summary stats for subject 11:
  choice_rate_overall: 0.82
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.045454545454545456
  error_rate_lvl_0_75: 0.3333333333333333
  error_rate_lvl_1_00: 0.55
  overall_rt_mean: 3.5400594028472825
  overall_rt_min: 0.6436090278625489
  rt_mean_correct_lvl_0_00: 1.4454840278625494
  rt_mean_correct_lvl_0_25: 1.6473590278625483
  rt_mean_correct_lvl_0_50: 2.241704265957774
  rt_mean_correct_lvl_0_75: 3.537775694529182
  rt_mean_correct_lvl_1_00: 2.423609027862538
============================================================
Drawing 500 posterior samples: 526it [00:00, 17163.29it/s]

------------------------------
Processing Subject 12/30
  True parameters for subject 12: {'v_norm': 0.7039211392402649, 'a_0': 0.9066845774650574, 'w_s_eff': 1.001580834388733, 't_0': 0.41090065240859985}
Observed summary stats for subject 12:
  choice_rate_overall: 0.72
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.038461538461538464
  error_rate_lvl_0_50: 0.21052631578947367
  error_rate_lvl_0_75: 0.5625
  error_rate_lvl_1_00: 0.6086956521739131
  overall_rt_mean: 3.978948469734188
  overall_rt_min: 0.5109006524085998
  rt_mean_correct_lvl_0_00: 1.0659006524086
  rt_mean_correct_lvl_0_25: 1.6425006524085937
  rt_mean_correct_lvl_0_50: 1.9829006524085895
  rt_mean_correct_lvl_0_75: 2.149472080980017
  rt_mean_correct_lvl_1_00: 1.6653450968530383
============================================================
Drawing 500 posterior samples: 581it [00:00, 17550.40it/s]

------------------------------
Processing Subject 13/30
  True parameters for subject 13: {'v_norm': 1.5679960250854492, 'a_0': 1.2167739868164062, 'w_s_eff': 0.4927939772605896, 't_0': 0.1328161507844925}
Observed summary stats for subject 13:
  choice_rate_overall: 0.34
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.38461538461538464
  error_rate_lvl_0_50: 0.7222222222222222
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 0.9642857142857143
  overall_rt_mean: 7.476357491266721
  overall_rt_min: 0.3628161507844926
  rt_mean_correct_lvl_0_00: 2.2938161507844765
  rt_mean_correct_lvl_0_25: 4.444066150784439
  rt_mean_correct_lvl_0_50: 0.7028161507844928
  rt_mean_correct_lvl_0_75: -999.0
  rt_mean_correct_lvl_1_00: 2.692816150784482
============================================================
WARNING: Skipping subject 13 due to invalid summary statistics in: ['rt_mean_correct_lvl_0_75']

------------------------------
Processing Subject 14/30
  True parameters for subject 14: {'v_norm': 0.5800482034683228, 'a_0': 1.7968698740005493, 'w_s_eff': 0.32126161456108093, 't_0': 0.4694765508174896}
Observed summary stats for subject 14:
  choice_rate_overall: 0.44
  error_rate_lvl_0_00: 0.19230769230769232
  error_rate_lvl_0_25: 0.3333333333333333
  error_rate_lvl_0_50: 0.5238095238095238
  error_rate_lvl_0_75: 0.9285714285714286
  error_rate_lvl_1_00: 0.9166666666666666
  overall_rt_mean: 7.177469682359683
  overall_rt_min: 0.9894765508174899
  rt_mean_correct_lvl_0_00: 3.686619407960318
  rt_mean_correct_lvl_0_25: 3.7804765508174554
  rt_mean_correct_lvl_0_50: 3.4494765508174603
  rt_mean_correct_lvl_0_75: 3.2394765508174745
  rt_mean_correct_lvl_1_00: 2.3944765508174823
============================================================
Drawing 500 posterior samples: 569it [00:00, 15673.72it/s]

------------------------------
Processing Subject 15/30
  True parameters for subject 15: {'v_norm': 1.2223037481307983, 'a_0': 1.9074561595916748, 'w_s_eff': 1.4688788652420044, 't_0': 0.206181138753891}
Observed summary stats for subject 15:
  choice_rate_overall: 0.52
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.3125
  error_rate_lvl_0_75: 0.8695652173913043
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.092914192152014
  overall_rt_min: 0.4361811387538911
  rt_mean_correct_lvl_0_00: 1.46555613875389
  rt_mean_correct_lvl_0_25: 2.168908411481154
  rt_mean_correct_lvl_0_50: 5.0098175023901925
  rt_mean_correct_lvl_0_75: 1.0061811387538915
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 15 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 16/30
  True parameters for subject 16: {'v_norm': 1.8328229188919067, 'a_0': 1.484946370124817, 'w_s_eff': 1.44927978515625, 't_0': 0.1856650859117508}
Observed summary stats for subject 16:
  choice_rate_overall: 0.56
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.043478260869565216
  error_rate_lvl_0_50: 0.5
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 5.368772448110576
  overall_rt_min: 0.4956650859117509
  rt_mean_correct_lvl_0_00: 1.1498958551425202
  rt_mean_correct_lvl_0_25: 2.174755995002649
  rt_mean_correct_lvl_0_50: 2.391915085911738
  rt_mean_correct_lvl_0_75: -999.0
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 16 due to invalid summary statistics in: ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 17/30
  True parameters for subject 17: {'v_norm': 0.5284128189086914, 'a_0': 0.5767345428466797, 'w_s_eff': 1.3454580307006836, 't_0': 0.4798676371574402}
Observed summary stats for subject 17:
  choice_rate_overall: 0.88
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.0
  error_rate_lvl_0_75: 0.19047619047619047
  error_rate_lvl_1_00: 0.3076923076923077
  overall_rt_mean: 2.4832835206985426
  overall_rt_min: 0.5398676371574402
  rt_mean_correct_lvl_0_00: 0.844534303824107
  rt_mean_correct_lvl_0_25: 0.9783676371574404
  rt_mean_correct_lvl_0_50: 1.7037565260463192
  rt_mean_correct_lvl_0_75: 2.5263382253927147
  rt_mean_correct_lvl_1_00: 1.2487565260463296
============================================================
Drawing 500 posterior samples: 565it [00:00, 16024.71it/s]

------------------------------
Processing Subject 18/30
  True parameters for subject 18: {'v_norm': 1.797640323638916, 'a_0': 1.759986400604248, 'w_s_eff': 0.7025091052055359, 't_0': 0.41824012994766235}
Observed summary stats for subject 18:
  choice_rate_overall: 0.34
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.47619047619047616
  error_rate_lvl_0_50: 0.7058823529411765
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.697401644182198
  overall_rt_min: 0.9182401299476626
  rt_mean_correct_lvl_0_00: 2.955462352169865
  rt_mean_correct_lvl_0_25: 4.210967402674895
  rt_mean_correct_lvl_0_50: 2.044240129947661
  rt_mean_correct_lvl_0_75: -999.0
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 18 due to invalid summary statistics in: ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 19/30
  True parameters for subject 19: {'v_norm': 1.1878613233566284, 'a_0': 1.0647002458572388, 'w_s_eff': 1.1261398792266846, 't_0': 0.2434977889060974}
Observed summary stats for subject 19:
  choice_rate_overall: 0.55
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.5
  error_rate_lvl_0_75: 0.6956521739130435
  error_rate_lvl_1_00: 0.9166666666666666
  overall_rt_mean: 5.661323783898345
  overall_rt_min: 0.43349778890609747
  rt_mean_correct_lvl_0_00: 1.372545407953713
  rt_mean_correct_lvl_0_25: 2.6290533444616315
  rt_mean_correct_lvl_0_50: 2.473497788906075
  rt_mean_correct_lvl_0_75: 3.0492120746203506
  rt_mean_correct_lvl_1_00: 0.6634977889060976
============================================================
Drawing 500 posterior samples: 596it [00:00, 18485.58it/s]

------------------------------
Processing Subject 20/30
  True parameters for subject 20: {'v_norm': 0.34597134590148926, 'a_0': 1.5689425468444824, 'w_s_eff': 0.4023526906967163, 't_0': 0.24121400713920593}
Observed summary stats for subject 20:
  choice_rate_overall: 0.57
  error_rate_lvl_0_00: 0.14285714285714285
  error_rate_lvl_0_25: 0.1875
  error_rate_lvl_0_50: 0.36
  error_rate_lvl_0_75: 0.5714285714285714
  error_rate_lvl_1_00: 0.7083333333333334
  overall_rt_mean: 6.259791984069329
  overall_rt_min: 0.5012140071392059
  rt_mean_correct_lvl_0_00: 3.167047340472509
  rt_mean_correct_lvl_0_25: 2.8019832379084146
  rt_mean_correct_lvl_0_50: 4.046839007139166
  rt_mean_correct_lvl_0_75: 3.4389917849169533
  rt_mean_correct_lvl_1_00: 3.6926425785677397
============================================================
Drawing 500 posterior samples: 539it [00:00, 18659.97it/s]

------------------------------
Processing Subject 21/30
  True parameters for subject 21: {'v_norm': 1.8654553890228271, 'a_0': 1.7020817995071411, 'w_s_eff': 0.24000394344329834, 't_0': 0.060143791139125824}
Observed summary stats for subject 21:
  choice_rate_overall: 0.23
  error_rate_lvl_0_00: 0.2916666666666667
  error_rate_lvl_0_25: 0.7333333333333333
  error_rate_lvl_0_50: 0.9523809523809523
  error_rate_lvl_0_75: 0.9565217391304348
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 8.52543307196199
  overall_rt_min: 0.6101437911391261
  rt_mean_correct_lvl_0_00: 4.284261438197895
  rt_mean_correct_lvl_0_25: 1.9876437911391194
  rt_mean_correct_lvl_0_50: 0.8801437911391263
  rt_mean_correct_lvl_0_75: 0.8801437911391263
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 21 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 22/30
  True parameters for subject 22: {'v_norm': 1.0476031303405762, 'a_0': 1.4955248832702637, 'w_s_eff': 1.1989580392837524, 't_0': 0.4862188994884491}
Observed summary stats for subject 22:
  choice_rate_overall: 0.56
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.34782608695652173
  error_rate_lvl_0_75: 0.7142857142857143
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 5.895382583713522
  overall_rt_min: 0.7362188994884491
  rt_mean_correct_lvl_0_00: 1.9028855661551123
  rt_mean_correct_lvl_0_25: 3.171843899488427
  rt_mean_correct_lvl_0_50: 3.152218899488427
  rt_mean_correct_lvl_0_75: 2.8862188994884233
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 22 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 23/30
  True parameters for subject 23: {'v_norm': 0.4632035791873932, 'a_0': 1.5425529479980469, 'w_s_eff': 0.6959190964698792, 't_0': 0.1819649189710617}
Observed summary stats for subject 23:
  choice_rate_overall: 0.59
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.34615384615384615
  error_rate_lvl_0_75: 0.7272727272727273
  error_rate_lvl_1_00: 0.8421052631578947
  overall_rt_mean: 5.696859302192913
  overall_rt_min: 0.5219649189710618
  rt_mean_correct_lvl_0_00: 2.3396921916983184
  rt_mean_correct_lvl_0_25: 3.174692191698303
  rt_mean_correct_lvl_0_50: 3.150200213088684
  rt_mean_correct_lvl_0_75: 2.445298252304378
  rt_mean_correct_lvl_1_00: 1.6886315856377188
============================================================
Drawing 500 posterior samples: 589it [00:00, 20204.67it/s]

------------------------------
Processing Subject 24/30
  True parameters for subject 24: {'v_norm': 0.5557202696800232, 'a_0': 1.9056270122528076, 'w_s_eff': 0.7499365210533142, 't_0': 0.2985593378543854}
Observed summary stats for subject 24:
  choice_rate_overall: 0.56
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.045454545454545456
  error_rate_lvl_0_50: 0.5555555555555556
  error_rate_lvl_0_75: 0.6875
  error_rate_lvl_1_00: 0.9565217391304348
  overall_rt_mean: 6.120293229198442
  overall_rt_min: 0.8085593378543856
  rt_mean_correct_lvl_0_00: 1.8785593378543803
  rt_mean_correct_lvl_0_25: 3.0471307664257914
  rt_mean_correct_lvl_0_50: 4.399809337854338
  rt_mean_correct_lvl_0_75: 5.784559337854313
  rt_mean_correct_lvl_1_00: 4.468559337854341
============================================================
Drawing 500 posterior samples: 533it [00:00, 14141.71it/s]

------------------------------
Processing Subject 25/30
  True parameters for subject 25: {'v_norm': 1.2634210586547852, 'a_0': 1.8499163389205933, 'w_s_eff': 0.8580634593963623, 't_0': 0.20220765471458435}
Observed summary stats for subject 25:
  choice_rate_overall: 0.4
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.2
  error_rate_lvl_0_50: 0.6875
  error_rate_lvl_0_75: 0.9545454545454546
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.275383061885822
  overall_rt_min: 0.7022076547145846
  rt_mean_correct_lvl_0_00: 2.6788743213812323
  rt_mean_correct_lvl_0_25: 3.479707654714555
  rt_mean_correct_lvl_0_50: 4.3542076547145365
  rt_mean_correct_lvl_0_75: 1.8722076547145856
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 25 due to invalid summary statistics in: ['rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 26/30
  True parameters for subject 26: {'v_norm': 1.7347384691238403, 'a_0': 1.903838872909546, 'w_s_eff': 1.254747986793518, 't_0': 0.4776397943496704}
Observed summary stats for subject 26:
  choice_rate_overall: 0.41
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.08695652173913043
  error_rate_lvl_0_50: 0.9285714285714286
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.0790323156833574
  overall_rt_min: 1.1776397943496708
  rt_mean_correct_lvl_0_00: 1.8555345311917735
  rt_mean_correct_lvl_0_25: 3.8247826514924954
  rt_mean_correct_lvl_0_50: 2.327639794349672
  rt_mean_correct_lvl_0_75: -999.0
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 26 due to invalid summary statistics in: ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']

------------------------------
Processing Subject 27/30
  True parameters for subject 27: {'v_norm': 0.6872667670249939, 'a_0': 1.3390897512435913, 'w_s_eff': 0.42004260420799255, 't_0': 0.06291325390338898}
Observed summary stats for subject 27:
  choice_rate_overall: 0.47
  error_rate_lvl_0_00: 0.0625
  error_rate_lvl_0_25: 0.4
  error_rate_lvl_0_50: 0.4583333333333333
  error_rate_lvl_0_75: 0.7222222222222222
  error_rate_lvl_1_00: 0.9090909090909091
  overall_rt_mean: 6.508669229334584
  overall_rt_min: 0.27291325390338905
  rt_mean_correct_lvl_0_00: 2.513579920570035
  rt_mean_correct_lvl_0_25: 2.0354132539033754
  rt_mean_correct_lvl_0_50: 3.023682484672593
  rt_mean_correct_lvl_0_75: 3.216913253903362
  rt_mean_correct_lvl_1_00: 1.6729132539033889
============================================================
Drawing 500 posterior samples: 574it [00:00, 18875.04it/s]

------------------------------
Processing Subject 28/30
  True parameters for subject 28: {'v_norm': 1.1660529375076294, 'a_0': 0.8532878756523132, 'w_s_eff': 1.195678472518921, 't_0': 0.41475656628608704}
Observed summary stats for subject 28:
  choice_rate_overall: 0.62
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.14285714285714285
  error_rate_lvl_0_75: 0.6666666666666666
  error_rate_lvl_1_00: 0.8461538461538461
  overall_rt_mean: 4.857549071097369
  overall_rt_min: 0.5647565662860871
  rt_mean_correct_lvl_0_00: 1.2242302504966123
  rt_mean_correct_lvl_0_25: 1.4607565662860833
  rt_mean_correct_lvl_0_50: 2.5564232329527345
  rt_mean_correct_lvl_0_75: 2.800470852000355
  rt_mean_correct_lvl_1_00: 0.7497565662860872
============================================================
Drawing 500 posterior samples: 553it [00:00, 8167.91it/s]

------------------------------
Processing Subject 29/30
  True parameters for subject 29: {'v_norm': 1.3147387504577637, 'a_0': 0.5381211042404175, 'w_s_eff': 0.8146315217018127, 't_0': 0.47111332416534424}
Observed summary stats for subject 29:
  choice_rate_overall: 0.71
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.13636363636363635
  error_rate_lvl_0_75: 0.5454545454545454
  error_rate_lvl_1_00: 0.875
  overall_rt_mean: 3.8095904601573927
  overall_rt_min: 0.5211133241653443
  rt_mean_correct_lvl_0_00: 1.0611133241653445
  rt_mean_correct_lvl_0_25: 1.8204883241653307
  rt_mean_correct_lvl_0_50: 1.2063764820600813
  rt_mean_correct_lvl_0_75: 1.1691133241653446
  rt_mean_correct_lvl_1_00: 0.8761133241653444
============================================================
Drawing 500 posterior samples: 527it [00:00, 12867.16it/s]

------------------------------
Processing Subject 30/30
  True parameters for subject 30: {'v_norm': 1.935542345046997, 'a_0': 1.3400617837905884, 'w_s_eff': 0.7027204036712646, 't_0': 0.13994838297367096}
Observed summary stats for subject 30:
  choice_rate_overall: 0.41
  error_rate_lvl_0_00: 0.04
  error_rate_lvl_0_25: 0.3333333333333333
  error_rate_lvl_0_50: 0.9545454545454546
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.775878837019198
  overall_rt_min: 0.329948382973671
  rt_mean_correct_lvl_0_00: 2.034531716306989
  rt_mean_correct_lvl_0_25: 2.3486983829736525
  rt_mean_correct_lvl_0_50: 1.1799483829736717
  rt_mean_correct_lvl_0_75: -999.0
  rt_mean_correct_lvl_1_00: -999.0
============================================================
WARNING: Skipping subject 30 due to invalid summary statistics in: ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']

============================================================
Finished all subject fits. Evaluating overall recovery...
  Parameter: v_norm
    R² (True vs. Posterior Mean): 0.861
    MAE: 0.132
    Bias (Mean of [Rec - True]): 0.045
  Parameter: a_0
    R² (True vs. Posterior Mean): 0.862
    MAE: 0.122
    Bias (Mean of [Rec - True]): -0.069
  Parameter: w_s_eff
    R² (True vs. Posterior Mean): 0.822
    MAE: 0.095
    Bias (Mean of [Rec - True]): 0.008
  Parameter: t_0
    R² (True vs. Posterior Mean): 0.805
    MAE: 0.050
    Bias (Mean of [Rec - True]): 0.009

Detailed parameter recovery results saved to minimal_nes_recovery_results\param_recovery_details_20250515_093844.csv
Recovery scatter plots saved to minimal_nes_recovery_results\param_recovery_scatter_20250515_093844.png

Parameter Recovery Script (NPE) finished.
Results in: minimal_nes_recovery_results
============================================================

Summary: 11 subject(s) were skipped due to invalid summary stats or errors:
{'subject_idx': 1, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 7, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 13, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_0_75']}
{'subject_idx': 15, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 16, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']}
{'subject_idx': 18, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']}
{'subject_idx': 21, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 22, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 25, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_1_00']}
{'subject_idx': 26, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']}
{'subject_idx': 30, 'reason': 'invalid_summary_stats', 'invalid_keys': ['rt_mean_correct_lvl_0_75', 'rt_mean_correct_lvl_1_00']}