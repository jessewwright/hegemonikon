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
 Neural network successfully converged after 191 epochs.NPE training took: 1305.61s

--- Running Recovery for 30 Synthetic Subjects ---

------------------------------
Processing Subject 1/30
  True parameters for subject 1: {'v_norm': 1.9354829788208008, 'a_0': 1.0922569036483765, 'w_s_eff': 1.4100663661956787, 't_0': 0.09445364773273468}
Observed summary stats for subject 1:
  choice_rate_overall: 0.53
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.4230769230769231
  error_rate_lvl_0_75: 0.9333333333333333
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 5.4792604332983466
  overall_rt_min: 0.3244536477327348
  rt_mean_correct_lvl_0_00: 0.685757995558822
  rt_mean_correct_lvl_0_25: 1.635167933447018
  rt_mean_correct_lvl_0_50: 2.5884536477327114
  rt_mean_correct_lvl_0_75: 0.4344536477327348
  rt_mean_correct_lvl_1_00: 5.4792604332983466
============================================================
Drawing 500 posterior samples: 555it [00:00, 13526.08it/s]

------------------------------
Processing Subject 2/30
  True parameters for subject 2: {'v_norm': 1.357511281967163, 'a_0': 1.8319886922836304, 'w_s_eff': 1.3761247396469116, 't_0': 0.21890252828598022}
Observed summary stats for subject 2:
  choice_rate_overall: 0.56
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.55
  error_rate_lvl_0_75: 0.9285714285714286
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 5.8194854158401395
  overall_rt_min: 0.5389025282859803
  rt_mean_correct_lvl_0_00: 1.6825025282859765
  rt_mean_correct_lvl_0_25: 3.127950147333572
  rt_mean_correct_lvl_0_50: 3.5411247505081698
  rt_mean_correct_lvl_0_75: 2.328902528285979
  rt_mean_correct_lvl_1_00: 5.8194854158401395
============================================================
Drawing 500 posterior samples: 543it [00:00, 17491.70it/s]

------------------------------
Processing Subject 3/30
  True parameters for subject 3: {'v_norm': 1.4791945219039917, 'a_0': 1.261195421218872, 'w_s_eff': 0.7515647411346436, 't_0': 0.41793742775917053}
Observed summary stats for subject 3:
  choice_rate_overall: 0.52
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.07692307692307693
  error_rate_lvl_0_50: 0.6923076923076923
  error_rate_lvl_0_75: 0.9411764705882353
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 5.998827462434762
  overall_rt_min: 0.6479374277591706
  rt_mean_correct_lvl_0_00: 2.0170678625417713
  rt_mean_correct_lvl_0_25: 2.298354094425829
  rt_mean_correct_lvl_0_50: 4.210437427759128
  rt_mean_correct_lvl_0_75: 1.4879374277591713
  rt_mean_correct_lvl_1_00: 5.998827462434762
============================================================
Drawing 500 posterior samples: 576it [00:00, 17520.37it/s]

------------------------------
Processing Subject 4/30
  True parameters for subject 4: {'v_norm': 1.6496338844299316, 'a_0': 1.6429975032806396, 'w_s_eff': 0.7613980770111084, 't_0': 0.12293700873851776}
Observed summary stats for subject 4:
  choice_rate_overall: 0.49
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.2
  error_rate_lvl_0_50: 0.7916666666666666
  error_rate_lvl_0_75: 0.9090909090909091
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.306339134281867
  overall_rt_min: 0.5129370087385179
  rt_mean_correct_lvl_0_00: 1.7929370087385124
  rt_mean_correct_lvl_0_25: 2.631270342071836
  rt_mean_correct_lvl_0_50: 4.262937008738474
  rt_mean_correct_lvl_0_75: 2.102937008738519
  rt_mean_correct_lvl_1_00: 6.306339134281867
============================================================
Drawing 500 posterior samples: 572it [00:00, 17587.99it/s]

------------------------------
Processing Subject 5/30
  True parameters for subject 5: {'v_norm': 0.31101030111312866, 'a_0': 1.3199280500411987, 'w_s_eff': 0.47503048181533813, 't_0': 0.2893492877483368}
Observed summary stats for subject 5:
  choice_rate_overall: 0.72
  error_rate_lvl_0_00: 0.05555555555555555
  error_rate_lvl_0_25: 0.125
  error_rate_lvl_0_50: 0.13636363636363635
  error_rate_lvl_0_75: 0.4375
  error_rate_lvl_1_00: 0.7
  overall_rt_mean: 4.7196314871787886
  overall_rt_min: 0.5693492877483368
  rt_mean_correct_lvl_0_00: 2.7364081112777314
  rt_mean_correct_lvl_0_25: 3.147920716319737
  rt_mean_correct_lvl_0_50: 2.043033498274645
  rt_mean_correct_lvl_0_75: 3.022682621081641
  rt_mean_correct_lvl_1_00: 2.2193492877483267
============================================================
Drawing 500 posterior samples: 593it [00:00, 19478.29it/s]

------------------------------
Processing Subject 6/30
  True parameters for subject 6: {'v_norm': 0.6463771462440491, 'a_0': 0.5476444959640503, 'w_s_eff': 0.5965905785560608, 't_0': 0.12829484045505524}
Observed summary stats for subject 6:
  choice_rate_overall: 0.76
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.15
  error_rate_lvl_0_50: 0.16
  error_rate_lvl_0_75: 0.21428571428571427
  error_rate_lvl_1_00: 0.6666666666666666
  overall_rt_mean: 3.3326040787458378
  overall_rt_min: 0.18829484045505523
  rt_mean_correct_lvl_0_00: 1.1062948404550486
  rt_mean_correct_lvl_0_25: 1.3988830757491666
  rt_mean_correct_lvl_0_50: 0.8325805547407686
  rt_mean_correct_lvl_0_75: 2.070113022273223
  rt_mean_correct_lvl_1_00: 1.0140091261693371
============================================================
Drawing 500 posterior samples: 579it [00:00, 19159.17it/s]

------------------------------
Processing Subject 7/30
  True parameters for subject 7: {'v_norm': 1.8542499542236328, 'a_0': 1.825695514678955, 'w_s_eff': 1.260512351989746, 't_0': 0.4592885375022888}
Observed summary stats for subject 7:
  choice_rate_overall: 0.41
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.047619047619047616
  error_rate_lvl_0_50: 0.8571428571428571
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.990208300375933
  overall_rt_min: 0.949288537502289
  rt_mean_correct_lvl_0_00: 1.9326218708356169
  rt_mean_correct_lvl_0_25: 2.8437885375022733
  rt_mean_correct_lvl_0_50: 5.785955204168886
  rt_mean_correct_lvl_0_75: 6.990208300375933
  rt_mean_correct_lvl_1_00: 6.990208300375933
============================================================
Drawing 500 posterior samples: 572it [00:00, 16436.98it/s]

------------------------------
Processing Subject 8/30
  True parameters for subject 8: {'v_norm': 1.60323965549469, 'a_0': 1.7359563112258911, 'w_s_eff': 0.4969750642776489, 't_0': 0.2963222861289978}
Observed summary stats for subject 8:
  choice_rate_overall: 0.34
  error_rate_lvl_0_00: 0.05555555555555555
  error_rate_lvl_0_25: 0.25
  error_rate_lvl_0_50: 0.8823529411764706
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.855449577283848
  overall_rt_min: 0.6263222861289979
  rt_mean_correct_lvl_0_00: 3.4904399331877887
  rt_mean_correct_lvl_0_25: 3.9836556194622927
  rt_mean_correct_lvl_0_50: 3.2263222861289793
  rt_mean_correct_lvl_0_75: 7.855449577283848
  rt_mean_correct_lvl_1_00: 7.855449577283848
============================================================
Drawing 500 posterior samples: 545it [00:00, 16698.90it/s]

------------------------------
Processing Subject 9/30
  True parameters for subject 9: {'v_norm': 0.1477012038230896, 'a_0': 1.5467503070831299, 'w_s_eff': 1.1507667303085327, 't_0': 0.4491327106952667}
Observed summary stats for subject 9:
  choice_rate_overall: 0.84
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.08
  error_rate_lvl_0_75: 0.22727272727272727
  error_rate_lvl_1_00: 0.6
  overall_rt_mean: 3.861071476984009
  overall_rt_min: 0.6491327106952668
  rt_mean_correct_lvl_0_00: 1.8965011317478946
  rt_mean_correct_lvl_0_25: 1.5396590264847378
  rt_mean_correct_lvl_0_50: 2.5956544498256857
  rt_mean_correct_lvl_0_75: 4.5891327106952176
  rt_mean_correct_lvl_1_00: 3.8507993773619007
============================================================
Drawing 500 posterior samples: 531it [00:00, 17395.32it/s]

------------------------------
Processing Subject 10/30
  True parameters for subject 10: {'v_norm': 1.8378281593322754, 'a_0': 1.3618494272232056, 'w_s_eff': 1.0837324857711792, 't_0': 0.43706995248794556}
Observed summary stats for subject 10:
  choice_rate_overall: 0.44
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.1875
  error_rate_lvl_0_50: 0.5789473684210527
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 0.9642857142857143
  overall_rt_mean: 6.376710779094696
  overall_rt_min: 0.7270699524879456
  rt_mean_correct_lvl_0_00: 1.8511608615788517
  rt_mean_correct_lvl_0_25: 1.9539930294110222
  rt_mean_correct_lvl_0_50: 1.2883199524879456
  rt_mean_correct_lvl_0_75: 6.376710779094696
  rt_mean_correct_lvl_1_00: 1.237069952487946
============================================================
Drawing 500 posterior samples: 565it [00:00, 15585.44it/s]

------------------------------
Processing Subject 11/30
  True parameters for subject 11: {'v_norm': 1.5392438173294067, 'a_0': 1.8810219764709473, 'w_s_eff': 0.28394389152526855, 't_0': 0.47048723697662354}
Observed summary stats for subject 11:
  choice_rate_overall: 0.25
  error_rate_lvl_0_00: 0.4166666666666667
  error_rate_lvl_0_25: 0.625
  error_rate_lvl_0_50: 0.8181818181818182
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 8.389421809244148
  overall_rt_min: 1.0504872369766238
  rt_mean_correct_lvl_0_00: 2.89548723697661
  rt_mean_correct_lvl_0_25: 3.8360427925321394
  rt_mean_correct_lvl_0_50: 6.94048723697653
  rt_mean_correct_lvl_0_75: 8.389421809244148
  rt_mean_correct_lvl_1_00: 8.389421809244148
============================================================
Drawing 500 posterior samples: 621it [00:00, 16581.02it/s]

------------------------------
Processing Subject 12/30
  True parameters for subject 12: {'v_norm': 1.75334632396698, 'a_0': 0.9282433986663818, 'w_s_eff': 0.9134774804115295, 't_0': 0.060360781848430634}
Observed summary stats for subject 12:
  choice_rate_overall: 0.5
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.25
  error_rate_lvl_0_50: 0.6666666666666666
  error_rate_lvl_0_75: 0.8823529411764706
  error_rate_lvl_1_00: 0.9
  overall_rt_mean: 5.602380390924213
  overall_rt_min: 0.1903607818484306
  rt_mean_correct_lvl_0_00: 1.1167607818484264
  rt_mean_correct_lvl_0_25: 1.406360781848424
  rt_mean_correct_lvl_0_50: 1.5553607818484225
  rt_mean_correct_lvl_0_75: 0.545360781848431
  rt_mean_correct_lvl_1_00: 0.40036078184843077
============================================================
Drawing 500 posterior samples: 552it [00:00, 19683.70it/s]

------------------------------
Processing Subject 13/30
  True parameters for subject 13: {'v_norm': 1.9021451473236084, 'a_0': 0.9132486581802368, 'w_s_eff': 1.0490400791168213, 't_0': 0.41876015067100525}
Observed summary stats for subject 13:
  choice_rate_overall: 0.47
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.6666666666666666
  error_rate_lvl_0_75: 0.9411764705882353
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 6.09901727081537
  overall_rt_min: 0.5787601506710053
  rt_mean_correct_lvl_0_00: 1.1432045951154501
  rt_mean_correct_lvl_0_25: 2.1177601506709953
  rt_mean_correct_lvl_0_50: 1.997510150670994
  rt_mean_correct_lvl_0_75: 0.9887601506710055
  rt_mean_correct_lvl_1_00: 6.09901727081537
============================================================
Drawing 500 posterior samples: 551it [00:00, 16117.20it/s]

------------------------------
Processing Subject 14/30
  True parameters for subject 14: {'v_norm': 1.39469313621521, 'a_0': 1.2984387874603271, 'w_s_eff': 0.8101781010627747, 't_0': 0.4104336202144623}
Observed summary stats for subject 14:
  choice_rate_overall: 0.48
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.125
  error_rate_lvl_0_50: 0.5555555555555556
  error_rate_lvl_0_75: 1.0
  error_rate_lvl_1_00: 0.9230769230769231
  overall_rt_mean: 6.254608137702935
  overall_rt_min: 0.7204336202144623
  rt_mean_correct_lvl_0_00: 1.818528858309696
  rt_mean_correct_lvl_0_25: 2.5540050487858723
  rt_mean_correct_lvl_0_50: 2.5629336202144444
  rt_mean_correct_lvl_0_75: 6.254608137702935
  rt_mean_correct_lvl_1_00: 0.7604336202144624
============================================================
Drawing 500 posterior samples: 556it [00:00, 18082.82it/s]

------------------------------
Processing Subject 15/30
  True parameters for subject 15: {'v_norm': 0.13710224628448486, 'a_0': 1.0995550155639648, 'w_s_eff': 0.826194703578949, 't_0': 0.15417706966400146}
Observed summary stats for subject 15:
  choice_rate_overall: 0.92
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.058823529411764705
  error_rate_lvl_0_75: 0.05263157894736842
  error_rate_lvl_1_00: 0.3
  overall_rt_mean: 2.718542904090869
  overall_rt_min: 0.27417706966400146
  rt_mean_correct_lvl_0_00: 1.7917961172830397
  rt_mean_correct_lvl_0_25: 1.6393944609683455
  rt_mean_correct_lvl_0_50: 2.0335520696639877
  rt_mean_correct_lvl_0_75: 2.9652881807750813
  rt_mean_correct_lvl_1_00: 2.186319926806848
============================================================
Drawing 500 posterior samples: 568it [00:00, 18465.65it/s]

------------------------------
Processing Subject 16/30
  True parameters for subject 16: {'v_norm': 1.1814574003219604, 'a_0': 1.9195231199264526, 'w_s_eff': 1.1386168003082275, 't_0': 0.4695746898651123}
Observed summary stats for subject 16:
  choice_rate_overall: 0.45
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.10526315789473684
  error_rate_lvl_0_50: 0.5
  error_rate_lvl_0_75: 0.92
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.121808610439286
  overall_rt_min: 1.2195746898651127
  rt_mean_correct_lvl_0_00: 2.4422062688124715
  rt_mean_correct_lvl_0_25: 3.8478099839827253
  rt_mean_correct_lvl_0_50: 6.015288975579324
  rt_mean_correct_lvl_0_75: 4.129574689865069
  rt_mean_correct_lvl_1_00: 7.121808610439286
============================================================
Drawing 500 posterior samples: 571it [00:00, 14627.33it/s]

------------------------------
Processing Subject 17/30
  True parameters for subject 17: {'v_norm': 0.1307877153158188, 'a_0': 1.8970260620117188, 'w_s_eff': 1.4548383951187134, 't_0': 0.06216118857264519}
Observed summary stats for subject 17:
  choice_rate_overall: 0.9
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.043478260869565216
  error_rate_lvl_0_75: 0.21739130434782608
  error_rate_lvl_1_00: 0.25
  overall_rt_mean: 3.0675450697153663
  overall_rt_min: 0.45216118857264537
  rt_mean_correct_lvl_0_00: 1.0471611885726457
  rt_mean_correct_lvl_0_25: 2.570911188572622
  rt_mean_correct_lvl_0_50: 2.4376157340271747
  rt_mean_correct_lvl_0_75: 2.85660563301707
  rt_mean_correct_lvl_1_00: 3.127994521905951
============================================================
Drawing 500 posterior samples: 540it [00:00, 20553.60it/s]

------------------------------
Processing Subject 18/30
  True parameters for subject 18: {'v_norm': 1.5643596649169922, 'a_0': 0.9124270677566528, 'w_s_eff': 1.1590030193328857, 't_0': 0.389026015996933}
Observed summary stats for subject 18:
  choice_rate_overall: 0.53
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.35
  error_rate_lvl_0_75: 0.7727272727272727
  error_rate_lvl_1_00: 0.9583333333333334
  overall_rt_mean: 5.66768378847837
  overall_rt_min: 0.519026015996933
  rt_mean_correct_lvl_0_00: 1.3763944370495615
  rt_mean_correct_lvl_0_25: 2.459026015996916
  rt_mean_correct_lvl_0_50: 2.1344106313815336
  rt_mean_correct_lvl_0_75: 0.9990260159969333
  rt_mean_correct_lvl_1_00: 0.9890260159969333
============================================================
Drawing 500 posterior samples: 593it [00:00, 17493.35it/s]

------------------------------
Processing Subject 19/30
  True parameters for subject 19: {'v_norm': 1.5036249160766602, 'a_0': 0.8879114389419556, 'w_s_eff': 0.7007095813751221, 't_0': 0.12624219059944153}
Observed summary stats for subject 19:
  choice_rate_overall: 0.55
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.45
  error_rate_lvl_0_75: 0.8260869565217391
  error_rate_lvl_1_00: 0.9444444444444444
  overall_rt_mean: 5.355033204829688
  overall_rt_min: 0.2262421905994415
  rt_mean_correct_lvl_0_00: 1.4236334949472613
  rt_mean_correct_lvl_0_25: 1.9568671905994304
  rt_mean_correct_lvl_0_50: 1.570787645144887
  rt_mean_correct_lvl_0_75: 0.6537421905994418
  rt_mean_correct_lvl_1_00: 1.5562421905994426
============================================================
Drawing 500 posterior samples: 582it [00:00, 18077.83it/s]

------------------------------
Processing Subject 20/30
  True parameters for subject 20: {'v_norm': 0.35686758160591125, 'a_0': 0.9274749159812927, 'w_s_eff': 0.5752167701721191, 't_0': 0.2890457808971405}
Observed summary stats for subject 20:
  choice_rate_overall: 0.69
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.375
  error_rate_lvl_0_75: 0.4
  error_rate_lvl_1_00: 0.5714285714285714
  overall_rt_mean: 4.623641588819018
  overall_rt_min: 0.3990457808971405
  rt_mean_correct_lvl_0_00: 2.264500326351672
  rt_mean_correct_lvl_0_25: 1.7881366899880442
  rt_mean_correct_lvl_0_50: 3.1903791142304443
  rt_mean_correct_lvl_0_75: 2.1846013364526815
  rt_mean_correct_lvl_1_00: 1.7165457808971343
============================================================
Drawing 500 posterior samples: 593it [00:00, 16792.17it/s]

------------------------------
Processing Subject 21/30
  True parameters for subject 21: {'v_norm': 1.7859132289886475, 'a_0': 1.3378312587738037, 'w_s_eff': 0.25801882147789, 't_0': 0.25313448905944824}
Observed summary stats for subject 21:
  choice_rate_overall: 0.3
  error_rate_lvl_0_00: 0.25
  error_rate_lvl_0_25: 0.375
  error_rate_lvl_0_50: 0.9090909090909091
  error_rate_lvl_0_75: 0.9230769230769231
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.78884034671783
  overall_rt_min: 0.6531344890594484
  rt_mean_correct_lvl_0_00: 2.7406344890594276
  rt_mean_correct_lvl_0_25: 2.544467822392764
  rt_mean_correct_lvl_0_50: 3.318134489059416
  rt_mean_correct_lvl_0_75: 1.1931344890594489
  rt_mean_correct_lvl_1_00: 7.78884034671783
============================================================
Drawing 500 posterior samples: 554it [00:00, 20470.83it/s]

------------------------------
Processing Subject 22/30
  True parameters for subject 22: {'v_norm': 0.8192044496536255, 'a_0': 1.4016838073730469, 'w_s_eff': 0.23707722127437592, 't_0': 0.4328218698501587}
Observed summary stats for subject 22:
  choice_rate_overall: 0.3
  error_rate_lvl_0_00: 0.4375
  error_rate_lvl_0_25: 0.5555555555555556
  error_rate_lvl_0_50: 0.6842105263157895
  error_rate_lvl_0_75: 0.7916666666666666
  error_rate_lvl_1_00: 0.9130434782608695
  overall_rt_mean: 8.01084656095504
  overall_rt_min: 0.8628218698501589
  rt_mean_correct_lvl_0_00: 2.2894885365168176
  rt_mean_correct_lvl_0_25: 4.582821869850115
  rt_mean_correct_lvl_0_50: 3.0744885365168044
  rt_mean_correct_lvl_0_75: 3.860821869850123
  rt_mean_correct_lvl_1_00: 3.0328218698501472
============================================================
Drawing 500 posterior samples: 567it [00:00, 15172.42it/s]

------------------------------
Processing Subject 23/30
  True parameters for subject 23: {'v_norm': 1.8294439315795898, 'a_0': 1.5824775695800781, 'w_s_eff': 0.6101105809211731, 't_0': 0.2053322196006775}
Observed summary stats for subject 23:
  choice_rate_overall: 0.31
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.5
  error_rate_lvl_0_50: 0.875
  error_rate_lvl_0_75: 0.9655172413793104
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.633052988076207
  overall_rt_min: 0.6453322196006777
  rt_mean_correct_lvl_0_00: 2.069776664045116
  rt_mean_correct_lvl_0_25: 3.1513322196006497
  rt_mean_correct_lvl_0_50: 1.6053322196006756
  rt_mean_correct_lvl_0_75: 1.3253322196006783
  rt_mean_correct_lvl_1_00: 7.633052988076207
============================================================
Drawing 500 posterior samples: 536it [00:00, 18998.16it/s]

------------------------------
Processing Subject 24/30
  True parameters for subject 24: {'v_norm': 1.9916819334030151, 'a_0': 0.7229750752449036, 'w_s_eff': 1.0644174814224243, 't_0': 0.19547751545906067}
Observed summary stats for subject 24:
  choice_rate_overall: 0.62
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.5555555555555556
  error_rate_lvl_0_75: 0.7894736842105263
  error_rate_lvl_1_00: 0.8666666666666667
  overall_rt_mean: 4.831996059584611
  overall_rt_min: 0.28547751545906064
  rt_mean_correct_lvl_0_00: 0.8989775154590612
  rt_mean_correct_lvl_0_25: 2.545834658316181
  rt_mean_correct_lvl_0_50: 1.1792275154590595
  rt_mean_correct_lvl_0_75: 0.802977515459061
  rt_mean_correct_lvl_1_00: 0.6454775154590608
============================================================
Drawing 500 posterior samples: 572it [00:00, 12932.19it/s]

------------------------------
Processing Subject 25/30
  True parameters for subject 25: {'v_norm': 0.34532368183135986, 'a_0': 1.371167778968811, 'w_s_eff': 0.37422311305999756, 't_0': 0.23014356195926666}
Observed summary stats for subject 25:
  choice_rate_overall: 0.66
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.10526315789473684
  error_rate_lvl_0_50: 0.5263157894736842
  error_rate_lvl_0_75: 0.38095238095238093
  error_rate_lvl_1_00: 0.7368421052631579
  overall_rt_mean: 5.248094750893102
  overall_rt_min: 0.5001435619592667
  rt_mean_correct_lvl_0_00: 2.5537799255956153
  rt_mean_correct_lvl_0_25: 3.55073179725335
  rt_mean_correct_lvl_0_50: 2.7656991175148034
  rt_mean_correct_lvl_0_75: 2.2809127927284836
  rt_mean_correct_lvl_1_00: 2.744143561959244
============================================================
Drawing 500 posterior samples: 592it [00:00, 15291.94it/s]

------------------------------
Processing Subject 26/30
  True parameters for subject 26: {'v_norm': 0.44914066791534424, 'a_0': 1.4610488414764404, 'w_s_eff': 0.8576701283454895, 't_0': 0.3169919550418854}
Observed summary stats for subject 26:
  choice_rate_overall: 0.61
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.3684210526315789
  error_rate_lvl_0_75: 0.5
  error_rate_lvl_1_00: 0.9090909090909091
  overall_rt_mean: 5.625265092575538
  overall_rt_min: 0.5869919550418854
  rt_mean_correct_lvl_0_00: 2.139719227769151
  rt_mean_correct_lvl_0_25: 2.7628252883751987
  rt_mean_correct_lvl_0_50: 3.1569919550418604
  rt_mean_correct_lvl_0_75: 3.14365862170853
  rt_mean_correct_lvl_1_00: 3.5369919550418585
============================================================
Drawing 500 posterior samples: 585it [00:00, 16029.40it/s]

------------------------------
Processing Subject 27/30
  True parameters for subject 27: {'v_norm': 1.6774263381958008, 'a_0': 1.1765766143798828, 'w_s_eff': 0.48077863454818726, 't_0': 0.42141661047935486}
Observed summary stats for subject 27:
  choice_rate_overall: 0.36
  error_rate_lvl_0_00: 0.047619047619047616
  error_rate_lvl_0_25: 0.45
  error_rate_lvl_0_50: 0.7777777777777778
  error_rate_lvl_0_75: 0.9565217391304348
  error_rate_lvl_1_00: 1.0
  overall_rt_mean: 7.302609979772562
  overall_rt_min: 0.5914166104793549
  rt_mean_correct_lvl_0_00: 2.449916610479339
  rt_mean_correct_lvl_0_25: 2.406871155933884
  rt_mean_correct_lvl_0_50: 3.4839166104793264
  rt_mean_correct_lvl_0_75: 0.851416610479355
  rt_mean_correct_lvl_1_00: 7.302609979772562
============================================================
Drawing 500 posterior samples: 554it [00:00, 17021.17it/s]

------------------------------
Processing Subject 28/30
  True parameters for subject 28: {'v_norm': 0.11069770902395248, 'a_0': 0.7415049076080322, 'w_s_eff': 1.2421633005142212, 't_0': 0.08463101834058762}
Observed summary stats for subject 28:
  choice_rate_overall: 0.89
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.0
  error_rate_lvl_0_50: 0.045454545454545456
  error_rate_lvl_0_75: 0.13636363636363635
  error_rate_lvl_1_00: 0.3181818181818182
  overall_rt_mean: 2.0013216063231205
  overall_rt_min: 0.1946310183405876
  rt_mean_correct_lvl_0_00: 0.6583152288669037
  rt_mean_correct_lvl_0_25: 0.6499643516739213
  rt_mean_correct_lvl_0_50: 1.4141548278643878
  rt_mean_correct_lvl_0_75: 1.1625257551826902
  rt_mean_correct_lvl_1_00: 1.0726310183405878
============================================================
Drawing 500 posterior samples: 577it [00:00, 18826.39it/s]

------------------------------
Processing Subject 29/30
  True parameters for subject 29: {'v_norm': 0.3164268732070923, 'a_0': 1.4169645309448242, 'w_s_eff': 0.6075246334075928, 't_0': 0.13220477104187012}
Observed summary stats for subject 29:
  choice_rate_overall: 0.82
  error_rate_lvl_0_00: 0.0
  error_rate_lvl_0_25: 0.07142857142857142
  error_rate_lvl_0_50: 0.13043478260869565
  error_rate_lvl_0_75: 0.3157894736842105
  error_rate_lvl_1_00: 0.6363636363636364
  overall_rt_mean: 3.6785079122543225
  overall_rt_min: 0.4422047710418702
  rt_mean_correct_lvl_0_00: 1.8232574026208108
  rt_mean_correct_lvl_0_25: 2.6156663095033865
  rt_mean_correct_lvl_0_50: 2.6262047710418543
  rt_mean_correct_lvl_0_75: 1.8952816941187858
  rt_mean_correct_lvl_1_00: 2.0097047710418687
============================================================
Drawing 500 posterior samples: 592it [00:00, 18136.21it/s]

------------------------------
Processing Subject 30/30
  True parameters for subject 30: {'v_norm': 0.9821648597717285, 'a_0': 1.4766957759857178, 'w_s_eff': 0.6558862924575806, 't_0': 0.2875348925590515}
Observed summary stats for subject 30:
  choice_rate_overall: 0.43
  error_rate_lvl_0_00: 0.05263157894736842
  error_rate_lvl_0_25: 0.26666666666666666
  error_rate_lvl_0_50: 0.6086956521739131
  error_rate_lvl_0_75: 0.8461538461538461
  error_rate_lvl_1_00: 0.9411764705882353
  overall_rt_mean: 6.674840003800386
  overall_rt_min: 0.5075348925590516
  rt_mean_correct_lvl_0_00: 2.0342015592257106
  rt_mean_correct_lvl_0_25: 2.738443983468126
  rt_mean_correct_lvl_0_50: 2.2953126703368145
  rt_mean_correct_lvl_0_75: 2.1100348925590415
  rt_mean_correct_lvl_1_00: 1.6475348925590525
============================================================
Drawing 500 posterior samples: 589it [00:00, 18618.03it/s]

============================================================
Finished all subject fits. Evaluating overall recovery...
  Parameter: v_norm
    R² (True vs. Posterior Mean): 0.959
    MAE: 0.106
    Bias (Mean of [Rec - True]): -0.010
  Parameter: a_0
    R² (True vs. Posterior Mean): 0.649
    MAE: 0.164
    Bias (Mean of [Rec - True]): -0.059
  Parameter: w_s_eff
    R² (True vs. Posterior Mean): 0.806
    MAE: 0.113
    Bias (Mean of [Rec - True]): -0.016
  Parameter: t_0
    R² (True vs. Posterior Mean): 0.629
    MAE: 0.065
    Bias (Mean of [Rec - True]): -0.001

Detailed parameter recovery results saved to minimal_nes_recovery_results\param_recovery_details_20250515_100710.csv
Recovery scatter plots saved to minimal_nes_recovery_results\param_recovery_scatter_20250515_100710.png

Parameter Recovery Script (NPE) finished.
Results in: minimal_nes_recovery_results
============================================================

No subjects were skipped due to invalid summary stats or errors.