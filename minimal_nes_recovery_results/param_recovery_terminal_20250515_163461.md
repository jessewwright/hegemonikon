(.venv_param_recovery) (base) PS C:\Users\jesse\Hegemonikon Project\hegemonikon> python run_parameter_recovery_minimal_nes_npe.py --n_subj 50 --n_trials 1000 --npe_train_sims 30000 --npe_posterior_samples 1000
WARNING (pytensor.configdefaults): g++ not available, if using conda: `conda install gxx`
WARNING (pytensor.configdefaults): g++ not detected!  PyTensor will be unable to compile C-implementations and will default to Python. Performance may be severely degraded. To remove this warning, set PyTensor flags cxx to an empty string.
Successfully imported sbi version: 0.24.0
Successfully imported arviz version: 0.21.0
Using device: cpu
============================================================
Starting Parameter Recovery for Minimal NES Parameters using NPE
============================================================

--- Training NPE for Minimal NES Recovery ---
Using 30000 simulations for training.
Using 30000 valid simulations for training NPE.
 Neural network successfully converged after 131 epochs.NPE training took: 22789.66s

--- Running Recovery for 50 Synthetic Subjects ---

------------------------------
Processing Subject 1/50
  True parameters for subject 1: {'v_norm': 1.1403995752334595, 'a_0': 1.4875223636627197, 'w_s_eff': 0.8972421884536743, 't_0': 0.1657380908727646}
Observed summary stats for subject 1:
  choice_rate_overall: 0.535
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0673076923076923
  error_rate_lvl_0_50: 0.47715736040609136
  error_rate_lvl_0_75: 0.8823529411764706
  error_rate_lvl_1_00: 0.9619565217391305
  overall_rt_mean: 5.955264140526047
  overall_rt_min: 0.3957380908727647
  rt_mean_correct_lvl_0_00: 1.857235675413819
  rt_mean_correct_lvl_0_25: 2.9508927300479972
  rt_mean_correct_lvl_0_50: 2.7255439161154604
  rt_mean_correct_lvl_0_75: 2.5009156704197184
  rt_mean_correct_lvl_1_00: 1.0843095194441932
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 26845.61it/s]

------------------------------
Processing Subject 2/50
  True parameters for subject 2: {'v_norm': 0.7597710490226746, 'a_0': 1.9293811321258545, 'w_s_eff': 0.600897490978241, 't_0': 0.2775462865829468}
Observed summary stats for subject 2:
  choice_rate_overall: 0.482
  error_rate_lvl_0_00: 0.03535353535353535
  error_rate_lvl_0_25: 0.25365853658536586
  error_rate_lvl_0_50: 0.5789473684210527
  error_rate_lvl_0_75: 0.8269230769230769
  error_rate_lvl_1_00: 0.9222222222222223
  overall_rt_mean: 6.907467310132964
  overall_rt_min: 0.6575462865829469
  rt_mean_correct_lvl_0_00: 3.2842478572635474
  rt_mean_correct_lvl_0_25: 3.928461319262647
  rt_mean_correct_lvl_0_50: 3.891069013855635
  rt_mean_correct_lvl_0_75: 3.478101842138471
  rt_mean_correct_lvl_1_00: 2.249689143725795
============================================================
Drawing 1000 posterior samples: 1067it [00:00, 19586.34it/s]

------------------------------
Processing Subject 3/50
  True parameters for subject 3: {'v_norm': 0.24043068289756775, 'a_0': 0.7891408801078796, 'w_s_eff': 1.42031729221344, 't_0': 0.34915047883987427}
Observed summary stats for subject 3:
  choice_rate_overall: 0.906
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.004878048780487805
  error_rate_lvl_0_75: 0.06091370558375635
  error_rate_lvl_1_00: 0.38028169014084506
  overall_rt_mean: 2.37589033382892
  overall_rt_min: 0.39915047883987426
  rt_mean_correct_lvl_0_00: 0.977500478839874
  rt_mean_correct_lvl_0_25: 1.125258586947981
  rt_mean_correct_lvl_0_50: 1.725915184722219
  rt_mean_correct_lvl_0_75: 2.1855829112722924
  rt_mean_correct_lvl_1_00: 2.0893777515671337
============================================================
Drawing 1000 posterior samples: 1050it [00:00, 18818.83it/s]

------------------------------
Processing Subject 4/50
  True parameters for subject 4: {'v_norm': 0.70476233959198, 'a_0': 1.1327557563781738, 'w_s_eff': 0.6585586667060852, 't_0': 0.3870692551136017}
Observed summary stats for subject 4:
  choice_rate_overall: 0.627
  error_rate_lvl_0_00: 0.02512562814070352
  error_rate_lvl_0_25: 0.0825242718446602
  error_rate_lvl_0_50: 0.32673267326732675
  error_rate_lvl_0_75: 0.6451612903225806
  error_rate_lvl_1_00: 0.7971014492753623
  overall_rt_mean: 5.363682422956217
  overall_rt_min: 0.5070692551136017
  rt_mean_correct_lvl_0_00: 2.2738218324331765
  rt_mean_correct_lvl_0_25: 2.691936979981307
  rt_mean_correct_lvl_0_50: 3.0250839609959286
  rt_mean_correct_lvl_0_75: 2.8185844066287307
  rt_mean_correct_lvl_1_00: 2.0558787789231148
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 28222.05it/s]

------------------------------
Processing Subject 5/50
  True parameters for subject 5: {'v_norm': 1.38494873046875, 'a_0': 0.9201749563217163, 'w_s_eff': 0.6033686995506287, 't_0': 0.07637161016464233}
Observed summary stats for subject 5:
  choice_rate_overall: 0.518
  error_rate_lvl_0_00: 0.00975609756097561
  error_rate_lvl_0_25: 0.10909090909090909
  error_rate_lvl_0_50: 0.6142131979695431
  error_rate_lvl_0_75: 0.8527918781725888
  error_rate_lvl_1_00: 0.9226519337016574
  overall_rt_mean: 5.745010494065279
  overall_rt_min: 0.18637161016464232
  rt_mean_correct_lvl_0_00: 1.5623814623813843
  rt_mean_correct_lvl_0_25: 2.1051471203687067
  rt_mean_correct_lvl_0_50: 1.881503189111999
  rt_mean_correct_lvl_0_75: 1.289819886026706
  rt_mean_correct_lvl_1_00: 1.0599430387360707
============================================================
Drawing 1000 posterior samples: 1099it [00:00, 17988.03it/s]

------------------------------
Processing Subject 6/50
  True parameters for subject 6: {'v_norm': 0.9363296627998352, 'a_0': 0.7481016516685486, 'w_s_eff': 1.1241427659988403, 't_0': 0.1049279123544693}
Observed summary stats for subject 6:
  choice_rate_overall: 0.708
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.008888888888888889
  error_rate_lvl_0_50: 0.20100502512562815
  error_rate_lvl_0_75: 0.5561797752808989
  error_rate_lvl_1_00: 0.7587939698492462
  overall_rt_mean: 3.8626189619469598
  overall_rt_min: 0.1649279123544693
  rt_mean_correct_lvl_0_00: 0.814425399791655
  rt_mean_correct_lvl_0_25: 1.4661386746862994
  rt_mean_correct_lvl_0_50: 1.8181354595242682
  rt_mean_correct_lvl_0_75: 1.412016519949398
  rt_mean_correct_lvl_1_00: 1.103469579021134
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 24008.06it/s]

------------------------------
Processing Subject 7/50
  True parameters for subject 7: {'v_norm': 1.7398260831832886, 'a_0': 1.9308292865753174, 'w_s_eff': 0.9095982909202576, 't_0': 0.2552528977394104}
Observed summary stats for subject 7:
  choice_rate_overall: 0.38
  error_rate_lvl_0_00: 0.005025125628140704
  error_rate_lvl_0_25: 0.2617801047120419
  error_rate_lvl_0_50: 0.7853403141361257
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.343496101140968
  overall_rt_min: 0.5352528977394104
  rt_mean_correct_lvl_0_00: 2.467525625012125
  rt_mean_correct_lvl_0_25: 3.5063167275266127
  rt_mean_correct_lvl_0_50: 3.9154968001783965
  rt_mean_correct_lvl_0_75: 7.343496101140968
  rt_mean_correct_lvl_1_00: 7.343496101140968
============================================================
Drawing 1000 posterior samples: 1072it [00:00, 18248.69it/s]

------------------------------
Processing Subject 8/50
  True parameters for subject 8: {'v_norm': 0.14580495655536652, 'a_0': 1.5253024101257324, 'w_s_eff': 0.9474306702613831, 't_0': 0.1919802576303482}
Observed summary stats for subject 8:
  choice_rate_overall: 0.812
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.03317535545023697
  error_rate_lvl_0_50: 0.06451612903225806
  error_rate_lvl_0_75: 0.2669902912621359
  error_rate_lvl_1_00: 0.5302325581395348
  overall_rt_mean: 4.030097969195827
  overall_rt_min: 0.39198025763034827
  rt_mean_correct_lvl_0_00: 1.7115956422457281
  rt_mean_correct_lvl_0_25: 2.309088100767589
  rt_mean_correct_lvl_0_50: 2.826807843837223
  rt_mean_correct_lvl_0_75: 3.557675621868724
  rt_mean_correct_lvl_1_00: 3.3510891685214057
============================================================
Drawing 1000 posterior samples: 1060it [00:00, 21389.83it/s]

------------------------------
Processing Subject 9/50
  True parameters for subject 9: {'v_norm': 1.0578101873397827, 'a_0': 0.8667804598808289, 'w_s_eff': 0.40438908338546753, 't_0': 0.11104445159435272}
Observed summary stats for subject 9:
  choice_rate_overall: 0.523
  error_rate_lvl_0_00: 0.045
  error_rate_lvl_0_25: 0.24401913875598086
  error_rate_lvl_0_50: 0.5
  error_rate_lvl_0_75: 0.7692307692307693
  error_rate_lvl_1_00: 0.8341013824884793
  overall_rt_mean: 5.68191624818384
  overall_rt_min: 0.18104445159435273
  rt_mean_correct_lvl_0_00: 1.6426674882435583
  rt_mean_correct_lvl_0_25: 1.965284957923451
  rt_mean_correct_lvl_0_50: 1.8916694515943409
  rt_mean_correct_lvl_0_75: 1.6636634992133925
  rt_mean_correct_lvl_1_00: 1.0049333404832397
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 28496.04it/s]

------------------------------
Processing Subject 10/50
  True parameters for subject 10: {'v_norm': 0.2614220380783081, 'a_0': 0.613544225692749, 'w_s_eff': 0.21628694236278534, 't_0': 0.4265623390674591}
Observed summary stats for subject 10:
  choice_rate_overall: 0.8
  error_rate_lvl_0_00: 0.08095238095238096
  error_rate_lvl_0_25: 0.1
  error_rate_lvl_0_50: 0.1989795918367347
  error_rate_lvl_0_75: 0.2702702702702703
  error_rate_lvl_1_00: 0.35406698564593303
  overall_rt_mean: 3.517089871253958
  overall_rt_min: 0.4665623390674591
  rt_mean_correct_lvl_0_00: 1.9709664841451673
  rt_mean_correct_lvl_0_25: 2.0705067835118895
  rt_mean_correct_lvl_0_50: 1.9371992817426065
  rt_mean_correct_lvl_0_75: 1.8270808575859672
  rt_mean_correct_lvl_1_00: 1.5793030798081926
============================================================
Drawing 1000 posterior samples: 1134it [00:00, 19599.39it/s]

------------------------------
Processing Subject 11/50
  True parameters for subject 11: {'v_norm': 1.9672348499298096, 'a_0': 0.7230873107910156, 'w_s_eff': 0.6330400109291077, 't_0': 0.19854721426963806}
Observed summary stats for subject 11:
  choice_rate_overall: 0.482
  error_rate_lvl_0_00: 0.005154639175257732
  error_rate_lvl_0_25: 0.18840579710144928
  error_rate_lvl_0_50: 0.5863874345549738
  error_rate_lvl_0_75: 0.8685446009389671
  error_rate_lvl_1_00: 0.9282051282051282
  overall_rt_mean: 5.888719757277962
  overall_rt_min: 0.26854721426963807
  rt_mean_correct_lvl_0_00: 1.2519669033888046
  rt_mean_correct_lvl_0_25: 1.9766424523648607
  rt_mean_correct_lvl_0_50: 1.3556358218645697
  rt_mean_correct_lvl_0_75: 0.7535472142696381
  rt_mean_correct_lvl_1_00: 0.48711864284106676
============================================================
Drawing 1000 posterior samples: 1044it [00:00, 18155.43it/s]

------------------------------
Processing Subject 12/50
  True parameters for subject 12: {'v_norm': 1.2023308277130127, 'a_0': 0.8573490381240845, 'w_s_eff': 0.4091845750808716, 't_0': 0.19318000972270966}
Observed summary stats for subject 12:
  choice_rate_overall: 0.486
  error_rate_lvl_0_00: 0.03125
  error_rate_lvl_0_25: 0.26666666666666666
  error_rate_lvl_0_50: 0.5287958115183246
  error_rate_lvl_0_75: 0.8153846153846154
  error_rate_lvl_1_00: 0.8634361233480177
  overall_rt_mean: 6.039285484725231
  overall_rt_min: 0.30318000972270964
  rt_mean_correct_lvl_0_00: 1.7415133430560332
  rt_mean_correct_lvl_0_25: 2.15073245727514
  rt_mean_correct_lvl_0_50: 2.091624454167139
  rt_mean_correct_lvl_0_75: 1.2334577875004853
  rt_mean_correct_lvl_1_00: 1.1341477516581915
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 27496.42it/s]

------------------------------
Processing Subject 13/50
  True parameters for subject 13: {'v_norm': 0.22344960272312164, 'a_0': 1.8087635040283203, 'w_s_eff': 0.4317202866077423, 't_0': 0.11167246103286743}
Observed summary stats for subject 13:
  choice_rate_overall: 0.636
  error_rate_lvl_0_00: 0.07407407407407407
  error_rate_lvl_0_25: 0.22674418604651161
  error_rate_lvl_0_50: 0.32857142857142857
  error_rate_lvl_0_75: 0.531578947368421
  error_rate_lvl_1_00: 0.6556603773584906
  overall_rt_mean: 5.902463685216881
  overall_rt_min: 0.3816724610328675
  rt_mean_correct_lvl_0_00: 3.341122461032835
  rt_mean_correct_lvl_0_25: 3.517612310656893
  rt_mean_correct_lvl_0_50: 3.8217433830186436
  rt_mean_correct_lvl_0_75: 3.909200550920466
  rt_mean_correct_lvl_1_00: 3.282357392539685
============================================================
Drawing 1000 posterior samples: 1048it [00:00, 20703.83it/s]

------------------------------
Processing Subject 14/50
  True parameters for subject 14: {'v_norm': 0.5355596542358398, 'a_0': 1.645125389099121, 'w_s_eff': 0.47083723545074463, 't_0': 0.4101320803165436}
Observed summary stats for subject 14:
  choice_rate_overall: 0.532
  error_rate_lvl_0_00: 0.08064516129032258
  error_rate_lvl_0_25: 0.2275132275132275
  error_rate_lvl_0_50: 0.42995169082125606
  error_rate_lvl_0_75: 0.6904761904761905
  error_rate_lvl_1_00: 0.8461538461538461
  overall_rt_mean: 6.514150266728386
  overall_rt_min: 0.6101320803165436
  rt_mean_correct_lvl_0_00: 3.164167168035818
  rt_mean_correct_lvl_0_25: 3.7645156419603443
  rt_mean_correct_lvl_0_50: 3.684030385401258
  rt_mean_correct_lvl_0_75: 3.152593618778057
  rt_mean_correct_lvl_1_00: 3.2445070803165157
============================================================
Drawing 1000 posterior samples: 1099it [00:00, 19560.13it/s]

------------------------------
Processing Subject 15/50
  True parameters for subject 15: {'v_norm': 1.5192729234695435, 'a_0': 1.0717406272888184, 'w_s_eff': 0.33964812755584717, 't_0': 0.3118003308773041}
Observed summary stats for subject 15:
  choice_rate_overall: 0.379
  error_rate_lvl_0_00: 0.07216494845360824
  error_rate_lvl_0_25: 0.37
  error_rate_lvl_0_50: 0.7414634146341463
  error_rate_lvl_0_75: 0.9175257731958762
  error_rate_lvl_1_00: 0.9806763285024155
  overall_rt_mean: 7.106482325402491
  overall_rt_min: 0.41180033087730405
  rt_mean_correct_lvl_0_00: 2.2542447753217347
  rt_mean_correct_lvl_0_25: 2.788784457861407
  rt_mean_correct_lvl_0_50: 2.1553852365376676
  rt_mean_correct_lvl_0_75: 1.3336753308773006
  rt_mean_correct_lvl_1_00: 0.9393003308773045
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 21941.21it/s]

------------------------------
Processing Subject 16/50
  True parameters for subject 16: {'v_norm': 0.8559433817863464, 'a_0': 1.8310554027557373, 'w_s_eff': 0.6394951343536377, 't_0': 0.37616127729415894}
Observed summary stats for subject 16:
  choice_rate_overall: 0.471
  error_rate_lvl_0_00: 0.029411764705882353
  error_rate_lvl_0_25: 0.21
  error_rate_lvl_0_50: 0.5906735751295337
  error_rate_lvl_0_75: 0.86
  error_rate_lvl_1_00: 0.9605911330049262
  overall_rt_mean: 6.946991961605535
  overall_rt_min: 0.686161277294159
  rt_mean_correct_lvl_0_00: 2.960908752041614
  rt_mean_correct_lvl_0_25: 4.128376467167535
  rt_mean_correct_lvl_0_50: 3.735022036787796
  rt_mean_correct_lvl_0_75: 3.4615184201512745
  rt_mean_correct_lvl_1_00: 3.307411277294138
============================================================
Drawing 1000 posterior samples: 1044it [00:00, 18861.69it/s]

------------------------------
Processing Subject 17/50
  True parameters for subject 17: {'v_norm': 0.9229609966278076, 'a_0': 1.3463668823242188, 'w_s_eff': 1.000031590461731, 't_0': 0.3608133792877197}
Observed summary stats for subject 17:
  choice_rate_overall: 0.614
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.055865921787709494
  error_rate_lvl_0_50: 0.328042328042328
  error_rate_lvl_0_75: 0.6635514018691588
  error_rate_lvl_1_00: 0.900523560209424
  overall_rt_mean: 5.328849414882651
  overall_rt_min: 0.5208133792877198
  rt_mean_correct_lvl_0_00: 1.701209855058642
  rt_mean_correct_lvl_0_25: 2.577499769820246
  rt_mean_correct_lvl_0_50: 3.3330181036971394
  rt_mean_correct_lvl_0_75: 2.6024800459543718
  rt_mean_correct_lvl_1_00: 1.9160765371824542
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 25310.04it/s]

------------------------------
Processing Subject 18/50
  True parameters for subject 18: {'v_norm': 0.14516396820545197, 'a_0': 1.740654468536377, 'w_s_eff': 1.0444855690002441, 't_0': 0.44999438524246216}
Observed summary stats for subject 18:
  choice_rate_overall: 0.775
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.03608247422680412
  error_rate_lvl_0_50: 0.08673469387755102
  error_rate_lvl_0_75: 0.35545023696682465
  error_rate_lvl_1_00: 0.6206896551724138
  overall_rt_mean: 4.680115648562891
  overall_rt_min: 0.7199943852424622
  rt_mean_correct_lvl_0_00: 2.180861732181231
  rt_mean_correct_lvl_0_25: 2.8405291446007337
  rt_mean_correct_lvl_0_50: 3.4210558377564015
  rt_mean_correct_lvl_0_75: 3.959553208771837
  rt_mean_correct_lvl_1_00: 4.163890489138525
============================================================
Drawing 1000 posterior samples: 1059it [00:00, 21028.81it/s]

------------------------------
Processing Subject 19/50
  True parameters for subject 19: {'v_norm': 1.0691208839416504, 'a_0': 0.740675151348114, 'w_s_eff': 0.9322901368141174, 't_0': 0.28912267088890076}
Observed summary stats for subject 19:
  choice_rate_overall: 0.648
  error_rate_lvl_0_00: 0.00510204081632653
  error_rate_lvl_0_25: 0.026455026455026454
  error_rate_lvl_0_50: 0.26291079812206575
  error_rate_lvl_0_75: 0.5970873786407767
  error_rate_lvl_1_00: 0.8520408163265306
  overall_rt_mean: 4.542261490736002
  overall_rt_min: 0.3191226708889008
  rt_mean_correct_lvl_0_00: 1.2434303631965906
  rt_mean_correct_lvl_0_25: 1.852437888280194
  rt_mean_correct_lvl_0_50: 1.8198233078315662
  rt_mean_correct_lvl_0_75: 1.5475564058286517
  rt_mean_correct_lvl_1_00: 0.8546399122682111
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 25844.50it/s]

------------------------------
Processing Subject 20/50
  True parameters for subject 20: {'v_norm': 1.852799415588379, 'a_0': 1.4653435945510864, 'w_s_eff': 0.9327043890953064, 't_0': 0.14249345660209656}
Observed summary stats for subject 20:
  choice_rate_overall: 0.43
  error_rate_lvl_0_00: 0.005025125628140704
  error_rate_lvl_0_25: 0.13963963963963963
  error_rate_lvl_0_50: 0.8241206030150754
  error_rate_lvl_0_75: 0.9702380952380952
  error_rate_lvl_1_00: 0.9952830188679245
  overall_rt_mean: 6.704372186338895
  overall_rt_min: 0.36249345660209664
  rt_mean_correct_lvl_0_00: 1.7194631535717884
  rt_mean_correct_lvl_0_25: 2.945896597963327
  rt_mean_correct_lvl_0_50: 2.575350599459222
  rt_mean_correct_lvl_0_75: 2.0764934566020887
  rt_mean_correct_lvl_1_00: 0.7324934566020969
============================================================
Drawing 1000 posterior samples: 1106it [00:00, 18394.90it/s]

------------------------------
Processing Subject 21/50
  True parameters for subject 21: {'v_norm': 0.7463917136192322, 'a_0': 0.7715263366699219, 'w_s_eff': 1.4303468465805054, 't_0': 0.3237290382385254}
Observed summary stats for subject 21:
  choice_rate_overall: 0.77
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.07614213197969544
  error_rate_lvl_0_75: 0.41798941798941797
  error_rate_lvl_1_00: 0.6834170854271356
  overall_rt_mean: 3.3874413594436605
  overall_rt_min: 0.3737290382385254
  rt_mean_correct_lvl_0_00: 0.9229984446312195
  rt_mean_correct_lvl_0_25: 1.2068923035446457
  rt_mean_correct_lvl_0_50: 1.8385092580187343
  rt_mean_correct_lvl_0_75: 2.088092674602148
  rt_mean_correct_lvl_1_00: 1.3405544350639176
============================================================
Drawing 1000 posterior samples: 1098it [00:00, 20329.06it/s]

------------------------------
Processing Subject 22/50
  True parameters for subject 22: {'v_norm': 1.7577179670333862, 'a_0': 1.03964364528656, 'w_s_eff': 0.26354438066482544, 't_0': 0.08676905930042267}
Observed summary stats for subject 22:
  choice_rate_overall: 0.311
  error_rate_lvl_0_00: 0.08994708994708994
  error_rate_lvl_0_25: 0.4739583333333333
  error_rate_lvl_0_50: 0.8585365853658536
  error_rate_lvl_0_75: 0.96875
  error_rate_lvl_1_00: 0.9864864864864865
  overall_rt_mean: 7.606145177442426
  overall_rt_min: 0.22676905930042265
  rt_mean_correct_lvl_0_00: 2.3009551058120317
  rt_mean_correct_lvl_0_25: 2.6793433167261402
  rt_mean_correct_lvl_0_50: 1.5284931972314524
  rt_mean_correct_lvl_0_75: 0.6667690593004231
  rt_mean_correct_lvl_1_00: 0.4801023926337562
============================================================
Drawing 1000 posterior samples: 1175it [00:00, 16246.43it/s]

------------------------------
Processing Subject 23/50
  True parameters for subject 23: {'v_norm': 1.9730515480041504, 'a_0': 0.8931703567504883, 'w_s_eff': 1.2533252239227295, 't_0': 0.21112163364887238}
Observed summary stats for subject 23:
  choice_rate_overall: 0.569
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.028037383177570093
  error_rate_lvl_0_50: 0.47715736040609136
  error_rate_lvl_0_75: 0.8333333333333334
  error_rate_lvl_1_00: 0.9789473684210527
  overall_rt_mean: 5.1463382095462045
  overall_rt_min: 0.2911216336488724
  rt_mean_correct_lvl_0_00: 1.013832744759983
  rt_mean_correct_lvl_0_25: 1.864631249033477
  rt_mean_correct_lvl_0_50: 1.8110245462702217
  rt_mean_correct_lvl_0_75: 1.1000871508902514
  rt_mean_correct_lvl_1_00: 0.4861216336488725
============================================================
Drawing 1000 posterior samples: 1093it [00:00, 19890.29it/s]

------------------------------
Processing Subject 24/50
  True parameters for subject 24: {'v_norm': 0.9785287976264954, 'a_0': 1.1174689531326294, 'w_s_eff': 0.4019448757171631, 't_0': 0.19295398890972137}
Observed summary stats for subject 24:
  choice_rate_overall: 0.508
  error_rate_lvl_0_00: 0.06310679611650485
  error_rate_lvl_0_25: 0.24401913875598086
  error_rate_lvl_0_50: 0.5576923076923077
  error_rate_lvl_0_75: 0.7613636363636364
  error_rate_lvl_1_00: 0.8855721393034826
  overall_rt_mean: 6.071590626366131
  overall_rt_min: 0.3229539889097214
  rt_mean_correct_lvl_0_00: 2.3176690148164396
  rt_mean_correct_lvl_0_25: 2.4615615838464118
  rt_mean_correct_lvl_0_50: 2.4134974671705725
  rt_mean_correct_lvl_0_75: 1.752477798433524
  rt_mean_correct_lvl_1_00: 0.8568670323879826
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 23513.57it/s]

------------------------------
Processing Subject 25/50
  True parameters for subject 25: {'v_norm': 0.5924942493438721, 'a_0': 1.9442089796066284, 'w_s_eff': 1.4640105962753296, 't_0': 0.1128883808851242}
Observed summary stats for subject 25:
  choice_rate_overall: 0.702
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.059907834101382486
  error_rate_lvl_0_75: 0.6091370558375635
  error_rate_lvl_1_00: 0.9269662921348315
  overall_rt_mean: 4.6842976433813455
  overall_rt_min: 0.45288838088512434
  rt_mean_correct_lvl_0_00: 1.3884275965713975
  rt_mean_correct_lvl_0_25: 2.243182498532172
  rt_mean_correct_lvl_0_50: 3.062937400492941
  rt_mean_correct_lvl_0_75: 3.954836432833134
  rt_mean_correct_lvl_1_00: 2.6221191501158785
============================================================
Drawing 1000 posterior samples: 1365it [00:00, 15978.50it/s]

------------------------------
Processing Subject 26/50
  True parameters for subject 26: {'v_norm': 1.4743378162384033, 'a_0': 1.601982831954956, 'w_s_eff': 0.2898311913013458, 't_0': 0.23190826177597046}
Observed summary stats for subject 26:
  choice_rate_overall: 0.294
  error_rate_lvl_0_00: 0.17391304347826086
  error_rate_lvl_0_25: 0.5980861244019139
  error_rate_lvl_0_50: 0.8306010928961749
  error_rate_lvl_0_75: 0.9754901960784313
  error_rate_lvl_1_00: 0.9847715736040609
  overall_rt_mean: 7.919191028962129
  overall_rt_min: 0.4319082617759705
  rt_mean_correct_lvl_0_00: 2.9478146945244847
  rt_mean_correct_lvl_0_25: 3.3066701665378466
  rt_mean_correct_lvl_0_50: 2.0654566488727357
  rt_mean_correct_lvl_0_75: 1.6559082617759664
  rt_mean_correct_lvl_1_00: 1.6819082617759638
============================================================
Drawing 1000 posterior samples: 1033it [00:00, 16897.15it/s]

------------------------------
Processing Subject 27/50
  True parameters for subject 27: {'v_norm': 1.1978085041046143, 'a_0': 1.451904535293579, 'w_s_eff': 1.486159086227417, 't_0': 0.2377908080816269}
Observed summary stats for subject 27:
  choice_rate_overall: 0.594
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.23834196891191708
  error_rate_lvl_0_75: 0.8216216216216217
  error_rate_lvl_1_00: 0.9629629629629629
  overall_rt_mean: 5.293757740000479
  overall_rt_min: 0.3677908080816269
  rt_mean_correct_lvl_0_00: 1.1723897385629098
  rt_mean_correct_lvl_0_25: 1.9931789359355
  rt_mean_correct_lvl_0_50: 3.2953418284897604
  rt_mean_correct_lvl_0_75: 2.491427171717972
  rt_mean_correct_lvl_1_00: 1.4227908080816278
============================================================
Drawing 1000 posterior samples: 1161it [00:00, 15894.93it/s]

------------------------------
Processing Subject 28/50
  True parameters for subject 28: {'v_norm': 0.22599872946739197, 'a_0': 1.091381549835205, 'w_s_eff': 0.9346702098846436, 't_0': 0.09372180700302124}
Observed summary stats for subject 28:
  choice_rate_overall: 0.843
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.014634146341463415
  error_rate_lvl_0_50: 0.06334841628959276
  error_rate_lvl_0_75: 0.2994652406417112
  error_rate_lvl_1_00: 0.48554913294797686
  overall_rt_mean: 3.2733074833035354
  overall_rt_min: 0.19372180700302122
  rt_mean_correct_lvl_0_00: 1.4225068537319896
  rt_mean_correct_lvl_0_25: 1.8563455693792463
  rt_mean_correct_lvl_0_50: 2.0339150437179834
  rt_mean_correct_lvl_0_75: 2.9393706619648237
  rt_mean_correct_lvl_1_00: 2.4475420317221004
============================================================
Drawing 1000 posterior samples: 1068it [00:00, 20274.63it/s]

------------------------------
Processing Subject 29/50
  True parameters for subject 29: {'v_norm': 0.7698345184326172, 'a_0': 0.9143738746643066, 'w_s_eff': 1.309112787246704, 't_0': 0.260348379611969}
Observed summary stats for subject 29:
  choice_rate_overall: 0.737
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.005291005291005291
  error_rate_lvl_0_50: 0.1210762331838565
  error_rate_lvl_0_75: 0.4488888888888889
  error_rate_lvl_1_00: 0.7701149425287356
  overall_rt_mean: 3.816946755774016
  overall_rt_min: 0.330348379611969
  rt_mean_correct_lvl_0_00: 1.056909226172814
  rt_mean_correct_lvl_0_25: 1.4475292306757954
  rt_mean_correct_lvl_0_50: 1.9736647061425683
  rt_mean_correct_lvl_0_75: 2.231154831224858
  rt_mean_correct_lvl_1_00: 1.2888483796119665
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 24068.54it/s]

------------------------------
Processing Subject 30/50
  True parameters for subject 30: {'v_norm': 0.3948206603527069, 'a_0': 1.1746737957000732, 'w_s_eff': 1.0999302864074707, 't_0': 0.22397544980049133}
Observed summary stats for subject 30:
  choice_rate_overall: 0.794
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.01932367149758454
  error_rate_lvl_0_50: 0.06878306878306878
  error_rate_lvl_0_75: 0.3160621761658031
  error_rate_lvl_1_00: 0.6095238095238096
  overall_rt_mean: 3.7041065071415806
  overall_rt_min: 0.35397544980049134
  rt_mean_correct_lvl_0_00: 1.3720849025368076
  rt_mean_correct_lvl_0_25: 1.7705764350221602
  rt_mean_correct_lvl_0_50: 2.5523277225277448
  rt_mean_correct_lvl_0_75: 2.8292784801034974
  rt_mean_correct_lvl_1_00: 2.2709266693126704
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 26348.61it/s]

------------------------------
Processing Subject 31/50
  True parameters for subject 31: {'v_norm': 1.6834968328475952, 'a_0': 1.9189131259918213, 'w_s_eff': 0.2560206949710846, 't_0': 0.32563671469688416}
Observed summary stats for subject 31:
  choice_rate_overall: 0.228
  error_rate_lvl_0_00: 0.2824074074074074
  error_rate_lvl_0_25: 0.7076923076923077
  error_rate_lvl_0_50: 0.9305555555555556
  error_rate_lvl_0_75: 0.9942528735632183
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 8.621629534236185
  overall_rt_min: 0.8156367146968844
  rt_mean_correct_lvl_0_00: 4.0237938971826726
  rt_mean_correct_lvl_0_25: 4.192654258556492
  rt_mean_correct_lvl_0_50: 2.488970048030209
  rt_mean_correct_lvl_0_75: 1.625636714696885
  rt_mean_correct_lvl_1_00: 8.621629534236185
============================================================
Drawing 1000 posterior samples: 1263it [00:00, 15597.03it/s]

------------------------------
Processing Subject 32/50
  True parameters for subject 32: {'v_norm': 1.4285080432891846, 'a_0': 1.1093534231185913, 'w_s_eff': 1.2756422758102417, 't_0': 0.3181098699569702}
Observed summary stats for subject 32:
  choice_rate_overall: 0.541
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.3191489361702128
  error_rate_lvl_0_75: 0.8382352941176471
  error_rate_lvl_1_00: 0.9421487603305785
  overall_rt_mean: 5.613857439646716
  overall_rt_min: 0.4181098699569702
  rt_mean_correct_lvl_0_00: 1.2769987588458578
  rt_mean_correct_lvl_0_25: 2.0832173968386796
  rt_mean_correct_lvl_0_50: 2.649672369956951
  rt_mean_correct_lvl_0_75: 1.5990189608660592
  rt_mean_correct_lvl_1_00: 1.0423955842426849
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 25282.58it/s]

------------------------------
Processing Subject 33/50
  True parameters for subject 33: {'v_norm': 1.87993323802948, 'a_0': 1.4632951021194458, 'w_s_eff': 1.0270546674728394, 't_0': 0.294094979763031}
Observed summary stats for subject 33:
  choice_rate_overall: 0.424
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.1164021164021164
  error_rate_lvl_0_50: 0.7581395348837209
  error_rate_lvl_0_75: 0.9664804469273743
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.799416271419519
  overall_rt_min: 0.5340949797630311
  rt_mean_correct_lvl_0_00: 1.7740447285067433
  rt_mean_correct_lvl_0_25: 3.200801566589352
  rt_mean_correct_lvl_0_50: 2.7723642105322415
  rt_mean_correct_lvl_0_75: 1.2807616464296983
  rt_mean_correct_lvl_1_00: 6.799416271419519
============================================================
Drawing 1000 posterior samples: 1081it [00:00, 17382.07it/s]

------------------------------
Processing Subject 34/50
  True parameters for subject 34: {'v_norm': 0.4397564232349396, 'a_0': 1.6028776168823242, 'w_s_eff': 1.3395514488220215, 't_0': 0.3149828314781189}
Observed summary stats for subject 34:
  choice_rate_overall: 0.763
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.004830917874396135
  error_rate_lvl_0_50: 0.08040201005025126
  error_rate_lvl_0_75: 0.3737864077669903
  error_rate_lvl_1_00: 0.7333333333333333
  overall_rt_mean: 4.316301900417792
  overall_rt_min: 0.534982831478119
  rt_mean_correct_lvl_0_00: 1.5694906034988425
  rt_mean_correct_lvl_0_25: 2.157264384876168
  rt_mean_correct_lvl_0_50: 3.0795183506037778
  rt_mean_correct_lvl_0_75: 3.6761456221757594
  rt_mean_correct_lvl_1_00: 3.1003674468627107
============================================================
Drawing 1000 posterior samples: 1098it [00:00, 22784.98it/s]

------------------------------
Processing Subject 35/50
  True parameters for subject 35: {'v_norm': 1.2194651365280151, 'a_0': 0.7255889177322388, 'w_s_eff': 0.48082637786865234, 't_0': 0.4387207627296448}
Observed summary stats for subject 35:
  choice_rate_overall: 0.547
  error_rate_lvl_0_00: 0.010101010101010102
  error_rate_lvl_0_25: 0.19796954314720813
  error_rate_lvl_0_50: 0.46113989637305697
  error_rate_lvl_0_75: 0.671957671957672
  error_rate_lvl_1_00: 0.8789237668161435
  overall_rt_mean: 5.555940257213109
  overall_rt_min: 0.5087207627296448
  rt_mean_correct_lvl_0_00: 1.7225472933418824
  rt_mean_correct_lvl_0_25: 2.4256827880460823
  rt_mean_correct_lvl_0_50: 1.9291053781142498
  rt_mean_correct_lvl_0_75: 1.2854949562780285
  rt_mean_correct_lvl_1_00: 0.9161281701370524
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 25081.65it/s]

------------------------------
Processing Subject 36/50
  True parameters for subject 36: {'v_norm': 1.1316101551055908, 'a_0': 1.0682648420333862, 'w_s_eff': 0.5483183860778809, 't_0': 0.39039647579193115}
Observed summary stats for subject 36:
  choice_rate_overall: 0.498
  error_rate_lvl_0_00: 0.02358490566037736
  error_rate_lvl_0_25: 0.17857142857142858
  error_rate_lvl_0_50: 0.5363128491620112
  error_rate_lvl_0_75: 0.8461538461538461
  error_rate_lvl_1_00: 0.9220183486238532
  overall_rt_mean: 6.262307444944374
  overall_rt_min: 0.5203964757919312
  rt_mean_correct_lvl_0_00: 2.260541403328151
  rt_mean_correct_lvl_0_25: 2.9244337428726532
  rt_mean_correct_lvl_0_50: 2.6984687649485384
  rt_mean_correct_lvl_0_75: 1.806729809125259
  rt_mean_correct_lvl_1_00: 1.4921611816742812
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 27625.18it/s]

------------------------------
Processing Subject 37/50
  True parameters for subject 37: {'v_norm': 0.7501102089881897, 'a_0': 0.7695863246917725, 'w_s_eff': 1.4430233240127563, 't_0': 0.24521395564079285}
Observed summary stats for subject 37:
  choice_rate_overall: 0.796
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.03902439024390244
  error_rate_lvl_0_75: 0.35638297872340424
  error_rate_lvl_1_00: 0.6972972972972973
  overall_rt_mean: 3.0846503086900667
  overall_rt_min: 0.29521395564079284
  rt_mean_correct_lvl_0_00: 0.8652647170621123
  rt_mean_correct_lvl_0_25: 1.0348139556407914
  rt_mean_correct_lvl_0_50: 1.7924220774681927
  rt_mean_correct_lvl_0_75: 1.8666189143184675
  rt_mean_correct_lvl_1_00: 1.1141425270693623
============================================================
Drawing 1000 posterior samples: 1088it [00:00, 20932.85it/s]

------------------------------
Processing Subject 38/50
  True parameters for subject 38: {'v_norm': 0.909160852432251, 'a_0': 1.3391644954681396, 'w_s_eff': 0.7150657176971436, 't_0': 0.23604728281497955}
Observed summary stats for subject 38:
  choice_rate_overall: 0.535
  error_rate_lvl_0_00: 0.01020408163265306
  error_rate_lvl_0_25: 0.11235955056179775
  error_rate_lvl_0_50: 0.4433497536945813
  error_rate_lvl_0_75: 0.7745098039215687
  error_rate_lvl_1_00: 0.8904109589041096
  overall_rt_mean: 6.077295296306003
  overall_rt_min: 0.4360472828149796
  rt_mean_correct_lvl_0_00: 2.1885730560108443
  rt_mean_correct_lvl_0_25: 2.928262472688373
  rt_mean_correct_lvl_0_50: 3.250206574850348
  rt_mean_correct_lvl_0_75: 2.893003804554087
  rt_mean_correct_lvl_1_00: 1.6539639494816427
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 26040.58it/s]

------------------------------
Processing Subject 39/50
  True parameters for subject 39: {'v_norm': 1.6781178712844849, 'a_0': 1.756341576576233, 'w_s_eff': 0.6731253266334534, 't_0': 0.38318875432014465}
Observed summary stats for subject 39:
  choice_rate_overall: 0.358
  error_rate_lvl_0_00: 0.022099447513812154
  error_rate_lvl_0_25: 0.3033175355450237
  error_rate_lvl_0_50: 0.8592964824120602
  error_rate_lvl_0_75: 0.9751243781094527
  error_rate_lvl_1_00: 0.9951923076923077
  overall_rt_mean: 7.606901574046602
  overall_rt_min: 0.7431887543201448
  rt_mean_correct_lvl_0_00: 2.9330192627947014
  rt_mean_correct_lvl_0_25: 3.8642091624833736
  rt_mean_correct_lvl_0_50: 3.0971173257486933
  rt_mean_correct_lvl_0_75: 2.3751887543201295
  rt_mean_correct_lvl_1_00: 1.123188754320145
============================================================
Drawing 1000 posterior samples: 1047it [00:00, 23690.11it/s]

------------------------------
Processing Subject 40/50
  True parameters for subject 40: {'v_norm': 1.6065278053283691, 'a_0': 1.7463575601577759, 'w_s_eff': 0.381409615278244, 't_0': 0.30860456824302673}
Observed summary stats for subject 40:
  choice_rate_overall: 0.311
  error_rate_lvl_0_00: 0.11261261261261261
  error_rate_lvl_0_25: 0.5613207547169812
  error_rate_lvl_0_50: 0.9050279329608939
  error_rate_lvl_0_75: 0.9779005524861878
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.938927416155329
  overall_rt_min: 0.5286045682430268
  rt_mean_correct_lvl_0_00: 3.203941600891513
  rt_mean_correct_lvl_0_25: 3.869787363941912
  rt_mean_correct_lvl_0_50: 2.873898685890064
  rt_mean_correct_lvl_0_75: 2.251104568243022
  rt_mean_correct_lvl_1_00: 7.938927416155329
============================================================
Drawing 1000 posterior samples: 1077it [00:00, 24422.40it/s]

------------------------------
Processing Subject 41/50
  True parameters for subject 41: {'v_norm': 0.16658714413642883, 'a_0': 0.5897418856620789, 'w_s_eff': 0.4315306842327118, 't_0': 0.3176167905330658}
Observed summary stats for subject 41:
  choice_rate_overall: 0.87
  error_rate_lvl_0_00: 0.02403846153846154
  error_rate_lvl_0_25: 0.03125
  error_rate_lvl_0_50: 0.11351351351351352
  error_rate_lvl_0_75: 0.20297029702970298
  error_rate_lvl_1_00: 0.2676056338028169
  overall_rt_mean: 2.792986607763759
  overall_rt_min: 0.3676167905330658
  rt_mean_correct_lvl_0_00: 1.707715312700544
  rt_mean_correct_lvl_0_25: 1.6296598012857457
  rt_mean_correct_lvl_0_50: 2.0268850832159777
  rt_mean_correct_lvl_0_75: 1.6363124427069722
  rt_mean_correct_lvl_1_00: 1.5855655084817748
============================================================
Drawing 1000 posterior samples: 1080it [00:00, 21046.64it/s]

------------------------------
Processing Subject 42/50
  True parameters for subject 42: {'v_norm': 0.7557377815246582, 'a_0': 1.0593963861465454, 'w_s_eff': 0.8252723217010498, 't_0': 0.06832313537597656}
Observed summary stats for subject 42:
  choice_rate_overall: 0.682
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.037037037037037035
  error_rate_lvl_0_50: 0.25
  error_rate_lvl_0_75: 0.535
  error_rate_lvl_1_00: 0.7958115183246073
  overall_rt_mean: 4.487766378326407
  overall_rt_min: 0.15832313537597656
  rt_mean_correct_lvl_0_00: 1.463137950190785
  rt_mean_correct_lvl_0_25: 2.0595731353759623
  rt_mean_correct_lvl_0_50: 2.4862316321079807
  rt_mean_correct_lvl_0_75: 1.78208657623618
  rt_mean_correct_lvl_1_00: 1.4542205712734086
============================================================
Drawing 1000 posterior samples: 1048it [00:00, 18441.92it/s]

------------------------------
Processing Subject 43/50
  True parameters for subject 43: {'v_norm': 0.9179653525352478, 'a_0': 0.7494379281997681, 'w_s_eff': 1.0361469984054565, 't_0': 0.4274609386920929}
Observed summary stats for subject 43:
  choice_rate_overall: 0.707
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.03
  error_rate_lvl_0_50: 0.16580310880829016
  error_rate_lvl_0_75: 0.5373134328358209
  error_rate_lvl_1_00: 0.7538461538461538
  overall_rt_mean: 4.136944883655304
  overall_rt_min: 0.4974609386920929
  rt_mean_correct_lvl_0_00: 1.1830533557537037
  rt_mean_correct_lvl_0_25: 1.7029764026096115
  rt_mean_correct_lvl_0_50: 2.2001317461454954
  rt_mean_correct_lvl_0_75: 2.2614394333157337
  rt_mean_correct_lvl_1_00: 1.3001692720254245
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 21643.89it/s]

------------------------------
Processing Subject 44/50
  True parameters for subject 44: {'v_norm': 1.4889273643493652, 'a_0': 0.5430973768234253, 'w_s_eff': 1.2669920921325684, 't_0': 0.4142661988735199}
Observed summary stats for subject 44:
  choice_rate_overall: 0.66
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.009478672985781991
  error_rate_lvl_0_50: 0.27225130890052357
  error_rate_lvl_0_75: 0.6063829787234043
  error_rate_lvl_1_00: 0.8686868686868687
  overall_rt_mean: 4.239835691256521
  overall_rt_min: 0.4542661988735199
  rt_mean_correct_lvl_0_00: 0.9263416705716323
  rt_mean_correct_lvl_0_25: 1.3632135672945689
  rt_mean_correct_lvl_0_50: 1.748654688082142
  rt_mean_correct_lvl_0_75: 1.307239171846489
  rt_mean_correct_lvl_1_00: 0.7208046604119815
============================================================
Drawing 1000 posterior samples: 1076it [00:00, 19973.58it/s]

------------------------------
Processing Subject 45/50
  True parameters for subject 45: {'v_norm': 0.24785619974136353, 'a_0': 1.8401212692260742, 'w_s_eff': 0.2979721128940582, 't_0': 0.3364911377429962}
Observed summary stats for subject 45:
  choice_rate_overall: 0.544
  error_rate_lvl_0_00: 0.21844660194174756
  error_rate_lvl_0_25: 0.29545454545454547
  error_rate_lvl_0_50: 0.41626794258373206
  error_rate_lvl_0_75: 0.6009852216748769
  error_rate_lvl_1_00: 0.7281553398058253
  overall_rt_mean: 6.703931178932169
  overall_rt_min: 0.6864911377429963
  rt_mean_correct_lvl_0_00: 3.9497209514075537
  rt_mean_correct_lvl_0_25: 4.048426621613923
  rt_mean_correct_lvl_0_50: 3.747064908234765
  rt_mean_correct_lvl_0_75: 4.158342989594806
  rt_mean_correct_lvl_1_00: 3.786669709171531
============================================================
Drawing 1000 posterior samples: 1048it [00:00, 18703.38it/s]

------------------------------
Processing Subject 46/50
  True parameters for subject 46: {'v_norm': 0.2577492594718933, 'a_0': 0.6260753870010376, 'w_s_eff': 0.6411905288696289, 't_0': 0.4174659848213196}
Observed summary stats for subject 46:
  choice_rate_overall: 0.888
  error_rate_lvl_0_00: 0.005
  error_rate_lvl_0_25: 0.015544041450777202
  error_rate_lvl_0_50: 0.10176991150442478
  error_rate_lvl_0_75: 0.16161616161616163
  error_rate_lvl_1_00: 0.2896174863387978
  overall_rt_mean: 2.5736497945213257
  overall_rt_min: 0.4774659848213196
  rt_mean_correct_lvl_0_00: 1.3751041757760898
  rt_mean_correct_lvl_0_25: 1.5097817742949988
  rt_mean_correct_lvl_0_50: 1.6506679552646608
  rt_mean_correct_lvl_0_75: 1.9471045390381758
  rt_mean_correct_lvl_1_00: 1.8064659848213092
============================================================
Drawing 1000 posterior samples: 1061it [00:00, 20761.56it/s]

------------------------------
Processing Subject 47/50
  True parameters for subject 47: {'v_norm': 0.4911583662033081, 'a_0': 0.9188604354858398, 'w_s_eff': 0.7887647747993469, 't_0': 0.22767791152000427}
Observed summary stats for subject 47:
  choice_rate_overall: 0.765
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.025
  error_rate_lvl_0_50: 0.16
  error_rate_lvl_0_75: 0.34210526315789475
  error_rate_lvl_1_00: 0.6363636363636364
  overall_rt_mean: 3.8127736023127947
  overall_rt_min: 0.3076779115200043
  rt_mean_correct_lvl_0_00: 1.451409254803581
  rt_mean_correct_lvl_0_25: 1.9698317576738382
  rt_mean_correct_lvl_0_50: 2.187499340091418
  rt_mean_correct_lvl_0_75: 2.1271979115199895
  rt_mean_correct_lvl_1_00: 2.020046332572624
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 26790.39it/s]

------------------------------
Processing Subject 48/50
  True parameters for subject 48: {'v_norm': 0.4080260396003723, 'a_0': 0.539776086807251, 'w_s_eff': 0.821948230266571, 't_0': 0.17236904799938202}
Observed summary stats for subject 48:
  choice_rate_overall: 0.863
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.010752688172043012
  error_rate_lvl_0_50: 0.0761904761904762
  error_rate_lvl_0_75: 0.205
  error_rate_lvl_1_00: 0.4126984126984127
  overall_rt_mean: 2.475749750327463
  overall_rt_min: 0.20236904799938202
  rt_mean_correct_lvl_0_00: 0.8732527689296135
  rt_mean_correct_lvl_0_25: 1.209530085781989
  rt_mean_correct_lvl_0_50: 1.5567898261024693
  rt_mean_correct_lvl_0_75: 1.507463387622013
  rt_mean_correct_lvl_1_00: 1.3850717507020762
============================================================
Drawing 1000 posterior samples: 1085it [00:00, 17544.52it/s]

------------------------------
Processing Subject 49/50
  True parameters for subject 49: {'v_norm': 0.8043027520179749, 'a_0': 1.0166130065917969, 'w_s_eff': 0.8936979174613953, 't_0': 0.14256659150123596}
Observed summary stats for subject 49:
  choice_rate_overall: 0.671
  error_rate_lvl_0_00: 0.0049504950495049506
  error_rate_lvl_0_25: 0.05472636815920398
  error_rate_lvl_0_50: 0.19444444444444445
  error_rate_lvl_0_75: 0.553921568627451
  error_rate_lvl_1_00: 0.7934272300469484
  overall_rt_mean: 4.518522182897322
  overall_rt_min: 0.26256659150123596
  rt_mean_correct_lvl_0_00: 1.4860491785659067
  rt_mean_correct_lvl_0_25: 1.830461328343331
  rt_mean_correct_lvl_0_50: 2.186014867363289
  rt_mean_correct_lvl_0_75: 2.270149009083636
  rt_mean_correct_lvl_1_00: 1.3291575005921417
============================================================
Drawing 1000 posterior samples: 100%|███████████████████████████████████████████| 1000/1000 [00:00<00:00, 26823.12it/s]

------------------------------
Processing Subject 50/50
  True parameters for subject 50: {'v_norm': 0.5280773043632507, 'a_0': 1.5956571102142334, 'w_s_eff': 0.26403531432151794, 't_0': 0.09947612881660461}
Observed summary stats for subject 50:
  choice_rate_overall: 0.503
  error_rate_lvl_0_00: 0.13930348258706468
  error_rate_lvl_0_25: 0.3062200956937799
  error_rate_lvl_0_50: 0.5191256830601093
  error_rate_lvl_0_75: 0.6923076923076923
  error_rate_lvl_1_00: 0.8177777777777778
  overall_rt_mean: 6.529076492794737
  overall_rt_min: 0.3194761288166047
  rt_mean_correct_lvl_0_00: 3.0902275739032823
  rt_mean_correct_lvl_0_25: 3.049889921920025
  rt_mean_correct_lvl_0_50: 3.3789079469983907
  rt_mean_correct_lvl_0_75: 3.146618985959431
  rt_mean_correct_lvl_1_00: 2.650695641011706
============================================================
Drawing 1000 posterior samples: 1055it [00:00, 18372.09it/s]

============================================================
Finished all subject fits. Evaluating overall recovery...
  Parameter: v_norm
    R² (True vs. Posterior Mean): 0.991
    MAE: 0.040
    Bias (Mean of [Rec - True]): 0.009
  Parameter: a_0
    R² (True vs. Posterior Mean): 0.979
    MAE: 0.047
    Bias (Mean of [Rec - True]): -0.019
  Parameter: w_s_eff
    R² (True vs. Posterior Mean): 0.981
    MAE: 0.041
    Bias (Mean of [Rec - True]): -0.019
  Parameter: t_0
    R² (True vs. Posterior Mean): 0.907
    MAE: 0.024
    Bias (Mean of [Rec - True]): 0.009

Detailed parameter recovery results saved to minimal_nes_recovery_results\param_recovery_details_20250515_163641.csv
Recovery scatter plots saved to minimal_nes_recovery_results\param_recovery_scatter_20250515_163641.png

Parameter Recovery Script (NPE) finished.
Results in: minimal_nes_recovery_results
============================================================

No subjects were skipped due to invalid summary stats or errors.