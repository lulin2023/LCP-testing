# LCP-testing
This repository contains the code to reproduce the experiments results in "Conditional testing based on localized conformal p-values" published at ICLR 2025 by Xiaoyang Wu, Lin Lu, Zhaojun Wang and Changliang Zou.

## Introduction 
We address conditional testing problems through the conformal inference framework. We define the localized conformal $p$-values by inverting prediction intervals and prove their theoretical properties. These defined $p$-values are then applied to several conditional testing problems to illustrate their practicality. Firstly, we propose a conditional outlier detection procedure to test for outliers in the conditional distribution with FDR control. We also introduce a novel conditional label screening problem with the goal of screening multivariate response variables and propose a screening procedure to control the FWER. Finally, we consider the two-sample conditional distribution test and define a weighted U-statistic through the aggregation of localized $p$-values.


## Folder contents
- `Synthetic-data codes/`: contains the codes for synthetic data simulation experiments in the paper.
- `Real-data codes/`: contains the codes for real data experiments in the paper, including three real data sets in application.


## Codes for reproducing results of the synthetic data.
- Conditional outlier detection `(Figure 1 in Section 4.1, Figure3-5 in Appendix E.1)`: `RLCP_outlier_ScenarioA_n.R/`, `RLCP_outlier_ScenarioA_alpha.R/` for Scenario A1, `RLCP_outlier_ScenarioB.R/` for Scenario B1. 
- Conditional label screening `(Figure 6 in Appendix E.2)`: `Label_screening_ScenarioA.R/`for Scenario A2.
- Two-sample conditional distribution test `(Figure 7 in Appendix E.3)`: `Scenario_A_simu_codes.R/` for Scenario A3, `Scenario_B_simu_codes.R/` for Scenario B3, `Scenario_C_simu_codes.R/` for Scenario C3.  
- Plot the results: `plots.R/`.


## Codes for reproducing results in Section 4.2 and Appendix E.4-E.5.
- Conditional outlier detection on House Sales data `(Table 2-3 in Appendix E.4)`: `HouseSales_codes.R/`.
- Conditional label screening on Health Indicator data `(Table 1 in Section 4.2)`: `Health_Indicators_label_screening.R/`.
- Two-sample conditional distribution test on airfoil data `(Table 4-5 in Appendix E.5)`: `case i.R/`, `case ii.R/` and `case iii.R/` .

## Citation
If you find this work useful, you can cite it with the following BibTex entry:

```bibtex
@inproceedings{
wu2025conditional,
title={Conditional Testing based on Localized Conformal \$p\$-values},
author={Xiaoyang Wu and Lin Lu and Zhaojun Wang and Changliang Zou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=Ip6UwB35uT}
}
```
