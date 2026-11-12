# Benchmarks

Each dataset is accompanied with a .json file describing its usage. For more details, see the source code in `emb/dataset.py`.

## Benchmarks used in the main text

### 15 benchmarks from GOLLuM [1]

| Benchmark                           | Path                                         |
| ----------------------------------- | -------------------------------------------- |
| Buchwald-Hartwig (BH1-5)            | buchwald-hartwig/bh_reaction_1.csv           |
| Additives screening (Additives 1-4) | additives/additive_rxn_screening_plate_1.csv |
| Suzuki-Miyaura                      | suzuki-miyaura/suzuki_miyaura_gollum.csv     |
| Redox                               | molecules/redox_mer_with_iupac.csv.gz        |
| Solvation                           | molecules/redox_mer_with_iupac.csv.gz        |
| Photoswitch                         | molecules/photoswitches.csv.gz               |
| Kinase                              | molecules/enamine10k.csv.gz                  |
| PCE                                 | molecules/photovoltaics_pce10k.csv.gz        |

### 3 benchmarks from our previous work [2]

| Benchmark                 | Path                                  |
| ------------------------- | ------------------------------------- |
| Shields                   | shields/shields_dataset.xlsx          |
| Buchwald-Hartwig (merged) | buchwald-hartwig/bh_reaction_all.xlsx |
| Suzuki-Miyaura (small)    | suzuki-miyaura/suzuki.csv             |

This can also be found in the [repo](https://github.com/chimie-paristech-CTM/HSF-ChemBO/tree/main/datasets) of our previous work.

## Benchmarks used in the SI

The following 19 benchmarks are used for LLM with prompt (procedure description). These are also provided in GOLLuM.

| Benchmark                           | Path                                         |
| ----------------------------------- | -------------------------------------------- |
| Buchwald-Hartwig (BH1-5)            | buchwald-hartwig/bh_reaction_1.csv           |
| Additives screening (Additives 1-4) | additives/additive_rxn_screening_plate_1.csv |
| Suzuki-Miyaura                      | suzuki-miyaura/suzuki_miyaura_gollum.csv     |
| Redox                               | molecules/redox_mer_with_iupac.csv.gz        |
| Solvation                           | molecules/redox_mer_with_iupac.csv.gz        |
| Photoswitch                         | molecules/photoswitches.csv.gz               |
| Kinase                              | molecules/enamine10k.csv.gz                  |
| PCE                                 | molecules/photovoltaics_pce10k.csv.gz        |
| C2 yield                            | c2-yield/c2_yield_data.csv                   |
| HPLC                                | hplc/hplc_data.csv                           |
| OER                                 | numeric/oer_data.csv                         |
| Vapdiff                             | vapdiff/vapdiff_data.csv                     |

## Other Benchmarks

GOLLuM used more benchmarks and most of them can be found in their GitHub [repo](https://github.com/schwallergroup/gollum/tree/main/data) as well.

## References

1. Ranković, B., Griffiths, RR. & Schwaller, P. Large language models as uncertainty-calibrated optimizers for experimental discovery. Nat Mach Intell 8 , 1466–1477 (2026). https://doi.org/10.1038/s42256-026-01283-z
2. Chen, G., Fleck, M. & Stuyver, T. Leveraging chemical hidden-space representations effectively in Bayesian optimization for experiment design through dimension-aware hyperpriors. J. Chem. Theory Comput. 9 June 2026; 22 (11): 5594–5608. https://doi.org/10.1021/acs.jctc.6c00251
