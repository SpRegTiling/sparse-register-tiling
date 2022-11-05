import numpy as np
import torch
import random
from typing import List
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.load import load_dense, load_csr, load_coo


CACHE_DIR = "/sdb/cache/workingset_size/"
os.makedirs(CACHE_DIR, exist_ok=True)

SS_CACHE_DIR = CACHE_DIR + "/ss"
os.makedirs(SS_CACHE_DIR, exist_ok=True)

DLMC_CACHE_DIR = CACHE_DIR + "/dlmc"
os.makedirs(DLMC_CACHE_DIR, exist_ok=True)

print(DLMC_CACHE_DIR)

BCOLS = 256

SUITE_SPARSE_MATRICES_PATHS = ['raefsky1/raefsky1.mtx', 'lpi_ex72a/lpi_ex72a.mtx', 'net100/net100.mtx', 'big/big.mtx', 'barth/barth.mtx', 'scagr7-2b/scagr7-2b.mtx', 'fpga_dcop_14/fpga_dcop_14.mtx', 'mplate/mplate.mtx', 'fpga_dcop_13/fpga_dcop_13.mtx', 'TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx', 'Ragusa18/Ragusa18.mtx', 'Ge87H76/Ge87H76.mtx', 'ns3Da/ns3Da.mtx', 'ABACUS_shell_ld/ABACUS_shell_ld.mtx', 'IG5-18/IG5-18.mtx', 'Si34H36/Si34H36.mtx', 'msc01440/msc01440.mtx', 'G11/G11.mtx', 'gen/gen.mtx', 'ibm32/ibm32.mtx', 'oscil_dcop_18/oscil_dcop_18.mtx', 'scsd8-2r/scsd8-2r.mtx', 'mc2depi/mc2depi.mtx', 'can_144/can_144.mtx', 'cr42/cr42.mtx', 'lp_grow22/lp_grow22.mtx', 'n4c6-b11/n4c6-b11.mtx', 'lp_80bau3b/lp_80bau3b.mtx', 'nos7/nos7.mtx', 'ch7-6-b3/ch7-6-b3.mtx', 'rajat22/rajat22.mtx', 'netscience/netscience.mtx', 'Goodwin_017/Goodwin_017.mtx', 'para-4/para-4.mtx', 'G10/G10.mtx', 'lutz30-23-b6/lutz30-23-b6.mtx', 'eris1176/eris1176.mtx', 'rel6/rel6.mtx', 'cis-n4c6-b4/cis-n4c6-b4.mtx', 'bibd_15_7/bibd_15_7.mtx', 'kineticBatchReactor_4/kineticBatchReactor_4.mtx', 'p0291/p0291.mtx', 'adder_dcop_37/adder_dcop_37.mtx', 'mycielskian5/mycielskian5.mtx', 'ok2010/ok2010.mtx', 'ct20stif/ct20stif.mtx', 'deter1/deter1.mtx', 'nemeth24/nemeth24.mtx', 'adder_dcop_48/adder_dcop_48.mtx', 'dwb512/dwb512.mtx', 'freeFlyingRobot_1/freeFlyingRobot_1.mtx', 'rajat01/rajat01.mtx', 'ldoor/ldoor.mtx', 'mcca/mcca.mtx', 'nemeth17/nemeth17.mtx', 'barrier2-9/barrier2-9.mtx', 'farm/farm.mtx', 'lowThrust_9/lowThrust_9.mtx', 'gre_216b/gre_216b.mtx', 'G46/G46.mtx', 'pde225/pde225.mtx', 'ex19/ex19.mtx', 'lhr02/lhr02.mtx', 'nasa2910/nasa2910.mtx', 'TSC_OPF_300/TSC_OPF_300.mtx', 'TS/TS.mtx', 'fe_body/fe_body.mtx', 'rgg_n_2_15_s0/rgg_n_2_15_s0.mtx', 'appu/appu.mtx', 'hvdc1/hvdc1.mtx', 'bips07_3078/bips07_3078.mtx', 'model6/model6.mtx', 'GD97_a/GD97_a.mtx', 'ash219/ash219.mtx', 'lowThrust_10/lowThrust_10.mtx', 'mycielskian3/mycielskian3.mtx', 'Erdos992/Erdos992.mtx', 'rosen7/rosen7.mtx', 'aa01/aa01.mtx', 'wang4/wang4.mtx', 'G61/G61.mtx', 'Goodwin_030/Goodwin_030.mtx', 'spaceStation_9/spaceStation_9.mtx', 'matrix-new_3/matrix-new_3.mtx', 'jnlbrng1/jnlbrng1.mtx', 'lp_scsd8/lp_scsd8.mtx', 'ABACUS_shell_hd/ABACUS_shell_hd.mtx', 'oscil_dcop_36/oscil_dcop_36.mtx', 'L-9/L-9.mtx', 'GD97_b/GD97_b.mtx', 'laminar_duct3D/laminar_duct3D.mtx', 'adder_dcop_40/adder_dcop_40.mtx', 'bcspwr10/bcspwr10.mtx', 'delaunay_n14/delaunay_n14.mtx', 'fpga_dcop_24/fpga_dcop_24.mtx', 'cis-n4c6-b15/cis-n4c6-b15.mtx', 'gams30a/gams30a.mtx', 'jan99jac080/jan99jac080.mtx', 'bibd_22_8/bibd_22_8.mtx', 'G37/G37.mtx', 'tube1/tube1.mtx', 'nv1/nv1.mtx', 'M10PI_n/M10PI_n.mtx', 'copter1/copter1.mtx', 'tub100/tub100.mtx', 'p2p-Gnutella06/p2p-Gnutella06.mtx', 'ASIC_100ks/ASIC_100ks.mtx', 'G2/G2.mtx', 'seymourl/seymourl.mtx', 'rajat21/rajat21.mtx', 'nxp1/nxp1.mtx', 'ga2010/ga2010.mtx', 'power9/power9.mtx', 'ch6-6-b2/ch6-6-b2.mtx', 'scrs8-2c/scrs8-2c.mtx', 'fpga_dcop_30/fpga_dcop_30.mtx', 'G51/G51.mtx', 'oscil_dcop_19/oscil_dcop_19.mtx', 'ex5/ex5.mtx', 'barth5/barth5.mtx', 'ex23/ex23.mtx', 'rail2586/rail2586.mtx', 'onetone2/onetone2.mtx', 'af_shell3/af_shell3.mtx', 'stufe-10/stufe-10.mtx', 'pds10/pds10.mtx', 'cavity02/cavity02.mtx', 'n2c6-b8/n2c6-b8.mtx', 'adder_dcop_68/adder_dcop_68.mtx', 'lp_ken_13/lp_ken_13.mtx', 'cond-mat-2003/cond-mat-2003.mtx', 'lp_share1b/lp_share1b.mtx', 'StocF-1465/StocF-1465.mtx', 'IG5-10/IG5-10.mtx', 'adder_dcop_60/adder_dcop_60.mtx', 'Tina_DisCog/Tina_DisCog.mtx', 'mark3jac140/mark3jac140.mtx', 'flower_7_1/flower_7_1.mtx', 'lnsp_131/lnsp_131.mtx', 'bibd_81_2/bibd_81_2.mtx', 'mhd3200b/mhd3200b.mtx', 'xingo_afonso_itaipu/xingo_afonso_itaipu.mtx', 'nc2010/nc2010.mtx', 'dataset12mfeatfactors_10NN/dataset12mfeatfactors_10NN.mtx', 'c-30/c-30.mtx', 'ASIC_680ks/ASIC_680ks.mtx', '2D_27628_bjtcai/2D_27628_bjtcai.mtx', 'mesh2e1/mesh2e1.mtx', 'adder_dcop_41/adder_dcop_41.mtx', 'sc205-2r/sc205-2r.mtx', 'TSC_OPF_1047/TSC_OPF_1047.mtx', 'dynamicSoaringProblem_8/dynamicSoaringProblem_8.mtx', 'lp_cre_c/lp_cre_c.mtx', 'gent113/gent113.mtx', 'G35/G35.mtx', 'wheel_7_1/wheel_7_1.mtx', 'rdb2048_noL/rdb2048_noL.mtx', 'swang1/swang1.mtx', 'windtunnel_evap3d/windtunnel_evap3d.mtx', 'bmw7st_1/bmw7st_1.mtx', 'adder_dcop_31/adder_dcop_31.mtx', 'Chebyshev3/Chebyshev3.mtx', 'c-26/c-26.mtx', 'beause/beause.mtx', 'mk12-b5/mk12-b5.mtx', 'bfwa782/bfwa782.mtx', 'reorientation_7/reorientation_7.mtx', 'CAG_mat364/CAG_mat364.mtx', 'reorientation_5/reorientation_5.mtx', 'delaunay_n17/delaunay_n17.mtx', 'shipsec1/shipsec1.mtx', 'adder_dcop_44/adder_dcop_44.mtx', 'adder_dcop_59/adder_dcop_59.mtx', 'GD95_a/GD95_a.mtx', 'bibd_9_3/bibd_9_3.mtx', 'cell2/cell2.mtx', 'Goodwin_010/Goodwin_010.mtx', 'GL7d19/GL7d19.mtx', 'yeast_30NN/yeast_30NN.mtx', 'Goodwin_127/Goodwin_127.mtx', 'reorientation_2/reorientation_2.mtx', 'nemscem/nemscem.mtx', 'p0040/p0040.mtx', 'wing/wing.mtx', 'hor_131/hor_131.mtx', 'std1_Jac2/std1_Jac2.mtx', 'af_shell5/af_shell5.mtx', 'ne2010/ne2010.mtx', 'jagmesh5/jagmesh5.mtx', 'rail_20209/rail_20209.mtx', 'cat_ears_3_1/cat_ears_3_1.mtx', 'adder_trans_02/adder_trans_02.mtx', 'ex21/ex21.mtx', 'iiasa/iiasa.mtx', 'kineticBatchReactor_6/kineticBatchReactor_6.mtx', 'nopoly/nopoly.mtx', 'brack2/brack2.mtx', 'G64/G64.mtx', 'bundle_adj/bundle_adj.mtx', 'rdist1/rdist1.mtx', 'rosen2/rosen2.mtx', 'conf6_0-8x8-80/conf6_0-8x8-80.mtx', 'ms2010/ms2010.mtx', 'fpga_dcop_07/fpga_dcop_07.mtx', 'lp_pilot4/lp_pilot4.mtx', 'onera_dual/onera_dual.mtx', 'msc10848/msc10848.mtx', 'GD06_theory/GD06_theory.mtx', 'smallworld/smallworld.mtx', 'relat3/relat3.mtx', 'plbuckle/plbuckle.mtx', 'ohne2/ohne2.mtx', 'TSOPF_RS_b300_c3/TSOPF_RS_b300_c3.mtx', 'Ecoli_10NN/Ecoli_10NN.mtx', 'af_2_k101/af_2_k101.mtx', 'FA/FA.mtx', 'G3_circuit/G3_circuit.mtx', 'TSOPF_RS_b39_c7/TSOPF_RS_b39_c7.mtx', 'GD99_c/GD99_c.mtx', 'barrier2-12/barrier2-12.mtx', 'TSOPF_FS_b39_c19/TSOPF_FS_b39_c19.mtx', 'fe_4elt2/fe_4elt2.mtx', 'nd6k/nd6k.mtx', 'IG5-15/IG5-15.mtx', 'oscil_dcop_14/oscil_dcop_14.mtx', 'plskz362/plskz362.mtx', 'G6/G6.mtx', 'ch7-6-b2/ch7-6-b2.mtx', 'fpga_dcop_21/fpga_dcop_21.mtx', 'boneS10/boneS10.mtx', 'adder_dcop_23/adder_dcop_23.mtx', 'shermanACd/shermanACd.mtx', 'cdde5/cdde5.mtx', 'ch6-6-b5/ch6-6-b5.mtx', 'G18/G18.mtx', 'oscil_dcop_50/oscil_dcop_50.mtx', 'Geo_1438/Geo_1438.mtx', 'can_24/can_24.mtx', 'Ge99H100/Ge99H100.mtx', 'hydr1/hydr1.mtx', 'big_dual/big_dual.mtx', 'adder_dcop_39/adder_dcop_39.mtx', 'thermal2/thermal2.mtx', 'mark3jac040sc/mark3jac040sc.mtx', 'ex3sta1/ex3sta1.mtx', 'optdigits_10NN/optdigits_10NN.mtx', 'fpga_dcop_36/fpga_dcop_36.mtx', 'IMDB/IMDB.mtx', 'deter3/deter3.mtx', 'Freescale2/Freescale2.mtx', 'ch7-8-b5/ch7-8-b5.mtx', 'kron_g500-logn17/kron_g500-logn17.mtx', 'ch8-8-b1/ch8-8-b1.mtx', 'olm2000/olm2000.mtx', 'jan99jac020/jan99jac020.mtx', 'pores_3/pores_3.mtx', 'EternityII_A/EternityII_A.mtx', 'c-42/c-42.mtx', 'Sandi_sandi/Sandi_sandi.mtx', 't2d_q9/t2d_q9.mtx', 'airfoil1_dual/airfoil1_dual.mtx', 'dgreen/dgreen.mtx', 'n4c5-b5/n4c5-b5.mtx', 'dwt_2680/dwt_2680.mtx', 'adder_dcop_36/adder_dcop_36.mtx', 'c-23/c-23.mtx', 'freeFlyingRobot_2/freeFlyingRobot_2.mtx', 'Dubcova1/Dubcova1.mtx', 'orbitRaising_2/orbitRaising_2.mtx', 'cz628/cz628.mtx', 'lock_700/lock_700.mtx', 'lp_pilot87/lp_pilot87.mtx', 'c-73b/c-73b.mtx', 'juba40k/juba40k.mtx', 'or2010/or2010.mtx', 'GD96_b/GD96_b.mtx', 'fpga_dcop_42/fpga_dcop_42.mtx', 'IG5-17/IG5-17.mtx', 'c8_mat11/c8_mat11.mtx', 'n4c6-b2/n4c6-b2.mtx', 'af_shell9/af_shell9.mtx', 'spaceStation_13/spaceStation_13.mtx', 'lp_pilot/lp_pilot.mtx', 'higgs-twitter/higgs-twitter.mtx', 'web-BerkStan/web-BerkStan.mtx', 'cat_ears_4_4/cat_ears_4_4.mtx', 'oscil_dcop_02/oscil_dcop_02.mtx', 'vanbody/vanbody.mtx', 'b_dyn/b_dyn.mtx', 'commanche_dual/commanche_dual.mtx', 'nd12k/nd12k.mtx', 'dermatology_5NN/dermatology_5NN.mtx', 'lpi_bgetam/lpi_bgetam.mtx', 'c-31/c-31.mtx', 'lp_degen3/lp_degen3.mtx', 'e40r0100/e40r0100.mtx', 'fpga_dcop_46/fpga_dcop_46.mtx', 'Delor338K/Delor338K.mtx', 'coPapersDBLP/coPapersDBLP.mtx', 'hydr1c/hydr1c.mtx', 'lp_fit2p/lp_fit2p.mtx', 'nemeth21/nemeth21.mtx', 'fv3/fv3.mtx', 'bayer10/bayer10.mtx', 'can_634/can_634.mtx', 'Kaufhold/Kaufhold.mtx', 'cage5/cage5.mtx', 's3rmq4m1/s3rmq4m1.mtx', 'lp_scagr7/lp_scagr7.mtx', 'lowThrust_13/lowThrust_13.mtx', 'well1850/well1850.mtx', 'air06/air06.mtx', 'rgg_n_2_19_s0/rgg_n_2_19_s0.mtx', 'fpga_dcop_04/fpga_dcop_04.mtx', 'sparsine/sparsine.mtx', 'human_gene2/human_gene2.mtx', 'dielFilterV3real/dielFilterV3real.mtx', 'lpi_ceria3d/lpi_ceria3d.mtx', 'Ga19As19H42/Ga19As19H42.mtx', 'dwt_209/dwt_209.mtx', 'Linux_call_graph/Linux_call_graph.mtx', 'wi2010/wi2010.mtx', 'bp_200/bp_200.mtx', 't2dal/t2dal.mtx', 'lp_pds_20/lp_pds_20.mtx', 'bfwb398/bfwb398.mtx', 'shyy41/shyy41.mtx', 'dynamicSoaringProblem_6/dynamicSoaringProblem_6.mtx', 'pwt/pwt.mtx', 'delaunay_n18/delaunay_n18.mtx', 'g7jac180/g7jac180.mtx', 'oscil_dcop_37/oscil_dcop_37.mtx', 'fem_hifreq_circuit/fem_hifreq_circuit.mtx', 'fpga_dcop_17/fpga_dcop_17.mtx', 'pwt/pwt.mtx', 'Alemdar/Alemdar.mtx', 'freeFlyingRobot_5/freeFlyingRobot_5.mtx', 'S20PI_n/S20PI_n.mtx', 'dawson5/dawson5.mtx', 'S10PI_n1/S10PI_n1.mtx', 'c-59/c-59.mtx', 'lp_25fv47/lp_25fv47.mtx', 'dw256B/dw256B.mtx', 'adder_dcop_29/adder_dcop_29.mtx', 'n3c6-b7/n3c6-b7.mtx', 'str_400/str_400.mtx', 'lpi_gran/lpi_gran.mtx', 'n3c4-b2/n3c4-b2.mtx', 'cond-mat/cond-mat.mtx', 'analytics/analytics.mtx', 'boyd1/boyd1.mtx', 'gas11/gas11.mtx', 'football/football.mtx', 'x104/x104.mtx', 'Franz3/Franz3.mtx', 'fpga_dcop_39/fpga_dcop_39.mtx', 'ex26/ex26.mtx', 'steam2/steam2.mtx', 'GD01_c/GD01_c.mtx', 'NACA0015/NACA0015.mtx', 'CurlCurl_2/CurlCurl_2.mtx', 'FEM_3D_thermal2/FEM_3D_thermal2.mtx', 'aircraft/aircraft.mtx', 'bloweya/bloweya.mtx', 'GD97_c/GD97_c.mtx', 'landmark/landmark.mtx', 'nv2/nv2.mtx', 'can_292/can_292.mtx', 'oscil_dcop_21/oscil_dcop_21.mtx', 'Ill_Stokes/Ill_Stokes.mtx', 'vsp_befref_fxm_2_4_air02/vsp_befref_fxm_2_4_air02.mtx', 'LargeRegFile/LargeRegFile.mtx', 'oscil_dcop_54/oscil_dcop_54.mtx', 'cnae9_10NN/cnae9_10NN.mtx', 'lpi_ex73a/lpi_ex73a.mtx', 'plantsmargin_12NN/plantsmargin_12NN.mtx', 'EX2/EX2.mtx', 'cis-n4c6-b2/cis-n4c6-b2.mtx', 'fpga_dcop_06/fpga_dcop_06.mtx', 'Hardesty2/Hardesty2.mtx', 'p0282/p0282.mtx', 'Sandi_authors/Sandi_authors.mtx', 'ACTIVSg10K/ACTIVSg10K.mtx', 'IG5-12/IG5-12.mtx', 'c-54/c-54.mtx', 'amazon-2008/amazon-2008.mtx', 'bibd_81_3/bibd_81_3.mtx', 'lshp3466/lshp3466.mtx', 'g7jac060sc/g7jac060sc.mtx', 'car4/car4.mtx', 'TF16/TF16.mtx', 'cis-n4c6-b1/cis-n4c6-b1.mtx', 'IG5-7/IG5-7.mtx', 'n3c4-b4/n3c4-b4.mtx', 'g7jac200/g7jac200.mtx', 'gupta3/gupta3.mtx', 'venkat50/venkat50.mtx', 'vsp_model1_crew1_cr42_south31/vsp_model1_crew1_cr42_south31.mtx', 'ut2010/ut2010.mtx', 'qiulp/qiulp.mtx', 'barth4/barth4.mtx', 'nemeth03/nemeth03.mtx', 'Chevron3/Chevron3.mtx', 'lp_sc50a/lp_sc50a.mtx', 'TSOPF_FS_b162_c4/TSOPF_FS_b162_c4.mtx', 'G44/G44.mtx', 'c-63/c-63.mtx', 'df2177/df2177.mtx', 'fpga_dcop_18/fpga_dcop_18.mtx', 'tols90/tols90.mtx', 'blckhole/blckhole.mtx', 'se/se.mtx', 'rel5/rel5.mtx', 'pwt/pwt.mtx', 'rajat06/rajat06.mtx', 'wheel_6_1/wheel_6_1.mtx', 'Goodwin_013/Goodwin_013.mtx', 'lp_sctap2/lp_sctap2.mtx', 'xingo3012/xingo3012.mtx', 'mk12-b1/mk12-b1.mtx', '1138_bus/1138_bus.mtx', 'wheel_5_1/wheel_5_1.mtx', 'c-29/c-29.mtx', 'lp_ship04l/lp_ship04l.mtx', 'gas_sensor/gas_sensor.mtx', 'adder_dcop_03/adder_dcop_03.mtx', 'mhda416/mhda416.mtx', 'ex27/ex27.mtx', 'G27/G27.mtx', 'pkustk10/pkustk10.mtx', 'fpga_dcop_28/fpga_dcop_28.mtx', 'ash608/ash608.mtx', 'p0033/p0033.mtx', 'vt2010/vt2010.mtx', '685_bus/685_bus.mtx', 'adder_dcop_06/adder_dcop_06.mtx', 'cage10/cage10.mtx', 'fpga_dcop_23/fpga_dcop_23.mtx', 'mark3jac020/mark3jac020.mtx', 'rajat14/rajat14.mtx', 'cz10228/cz10228.mtx', 'bbmat/bbmat.mtx', 'rajat29/rajat29.mtx', 'WorldCities/WorldCities.mtx', 'dataset16mfeatkarhunen_10NN/dataset16mfeatkarhunen_10NN.mtx', 'ch7-6-b4/ch7-6-b4.mtx', 'bwm200/bwm200.mtx', 'TF10/TF10.mtx', 'hi2010/hi2010.mtx', 'c-53/c-53.mtx', 'c-60/c-60.mtx', 'SiH4/SiH4.mtx', 'ch6-6-b1/ch6-6-b1.mtx', 'relat5/relat5.mtx', 'can_73/can_73.mtx', 'G3/G3.mtx', 'goddardRocketProblem_1/goddardRocketProblem_1.mtx', 'fpga_dcop_03/fpga_dcop_03.mtx', 'pkustk07/pkustk07.mtx', 'soc-sign-Slashdot090221/soc-sign-Slashdot090221.mtx', 'adder_dcop_04/adder_dcop_04.mtx', 'cavity20/cavity20.mtx', 'nnc1374/nnc1374.mtx', 'l30/l30.mtx', 'F2/F2.mtx', 'cavity23/cavity23.mtx', 'crack_dual/crack_dual.mtx', 'Tina_AskCog/Tina_AskCog.mtx', 'garon1/garon1.mtx', 'G13/G13.mtx', 'Chevron4/Chevron4.mtx', 'west0167/west0167.mtx', 'adder_dcop_53/adder_dcop_53.mtx', 'wiki-Talk/wiki-Talk.mtx', 'ks2010/ks2010.mtx', 'FEM_3D_thermal1/FEM_3D_thermal1.mtx', 'adder_dcop_51/adder_dcop_51.mtx', 'jagmesh9/jagmesh9.mtx', 'will57/will57.mtx', 'whitaker3/whitaker3.mtx', 'fullb/fullb.mtx', 't3dl_e/t3dl_e.mtx', 's2rmt3m1/s2rmt3m1.mtx', 'ch8-8-b4/ch8-8-b4.mtx', 'bibd_49_3/bibd_49_3.mtx', 'g7jac140/g7jac140.mtx']


if __name__ == "__main__":
    random.seed(10)
    SS_MATRICES_TO_PLOT = 507
    SS_MATRICES_TO_SELECT_FROM = 5000
    # Matrices that consume too much ram for our current setup
    SS_SKIP_LIST = ["mc2depi", "wikipedia-20051105", "circuit5M_dc", "memchip", "mycielskian19", "mycielskian20", "333SP", "ss"]
    MAX_SS_MATRIX_SIZE = 4e6

    # import ssgetpy
    # matrix_list = ssgetpy.search(rowbounds=[0, MAX_SS_MATRIX_SIZE],
    #                              colbounds=[0, MAX_SS_MATRIX_SIZE],
    #                              limit=SS_MATRICES_TO_SELECT_FROM)
    # matrix_ids = [matrix.id for matrix in matrix_list]
    # random.shuffle(matrix_ids)
    #
    # ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids[:SS_MATRICES_TO_PLOT], skip_list=SS_SKIP_LIST, loader=load_csr)
    #print(matrix_ids[:SS_MATRICES_TO_PLOT])
    PLOT_DIR = SCRIPT_DIR + "/../plots/"

    torch.set_grad_enabled(False)
    tile_shapes = [(x, x) for x in range(16, 512, 4)]

    #
    # BUCKET_SIZE = 0.05
    # num_buckets = int(1 / BUCKET_SIZE) + 1

    def run(loader, name, cache_dir, recompute=False):
        variation_per_tilesize = []
        tile_sizes = []
        matrix_density = []

        tile_sizes = np.array([x[0] for x in tile_shapes])
        np.save(cache_dir + "/tile_sizes.npy", tile_sizes)

        for matrix, path in loader:
            print(path)
            filepath = "/".join(path.split("/")[4:-1])
            filename = os.path.basename(path).split('.')[0]
            print(filepath, matrix.shape, end=' ')

            cache_dir_tmp = cache_dir + f"/{filepath}"
            os.makedirs(cache_dir_tmp, exist_ok=True)

            if os.path.exists(cache_dir_tmp + f"/{filename}_working_set_sizes.npy"):
                print("skipped")
                continue
            else:
                print("computing")

            for tile_shape in tile_shapes:
                (tile_rows, tile_cols) = tile_shape

                num_tiles, num_empty_tiles, active_rows, active_cols, densities = \
                    torch.ops.spmm_benchmarks.tile_stats_csr_not_binned(tile_rows, tile_cols, matrix)

                assert num_empty_tiles >= 0 and num_empty_tiles <= num_tiles

                active_rows = active_rows.numpy()
                active_cols = active_cols.numpy()
                densities = densities.numpy()

                # Assume CSR storage
                nnz = densities * tile_rows * tile_cols
                working_set_sizes = nnz * 2 + tile_rows \
                                    + BCOLS * active_rows * tile_rows \
                                    + BCOLS * active_cols * tile_cols

                np.save(cache_dir_tmp + f"/{filename}_working_set_sizes.npy", working_set_sizes)
                del working_set_sizes
                np.save(cache_dir_tmp + f"/{filename}_active_rows.npy", active_rows)
                del active_rows
                np.save(cache_dir_tmp + f"/{filename}_active_cols.npy", active_cols)
                del active_cols
                np.save(cache_dir_tmp + f"/{filename}_densities.npy", densities)
                del densities
            del matrix


    # dlmc_loader = islice(DLMCLoader(loader=load_csr, models=["transformer"],
    #                                 pruning_methods=["magnitude_pruning"], sparsities=[0.7]), 1)
    # dlmc_loader = DLMCLoader(loader=load_csr)
    # run(dlmc_loader, "ml", cache_dir=DLMC_CACHE_DIR, recompute=False)

    # ss_loader = islice(SuiteSparseLoader(matrix_ids=matrix_ids[:SS_MATRICES_TO_PLOT],
    #                                      skip_list=SS_SKIP_LIST, loader=load_csr), 1)
    ss_loader = SuiteSparseLoader(paths=SUITE_SPARSE_MATRICES_PATHS, skip_list=SS_SKIP_LIST, loader=load_csr)
    run(ss_loader, "ss", cache_dir=SS_CACHE_DIR, recompute=False)

