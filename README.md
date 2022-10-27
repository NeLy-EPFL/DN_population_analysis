# DN_population_analysis

This repository hold the code for the [Aymanns et al., 2022](https://elifesciences.org/articles/81527).

It is split into two parts, the [preprocessing](DN_population_analysis/preprocessing) and the [analysis](DN_population_analysis/analysis).

The data preprocessed data can be found on the [here](https://dataverse.harvard.edu/dataverse/DNs).

Each 9min recording is saved in a different folder.
You can specify the recordings you want to use in [recordings.txt](recordings.txt).
The file is read in a way that support commenting out lines using '#'.

If you reuse any of the code, please cite our [publication](https://elifesciences.org/articles/81527).
```
@article {aymanns2022
article_type = {journal},
title = {Descending neuron population dynamics during odor-evoked and spontaneous limb-dependent behaviors},
author = {Aymanns, Florian and Chen, Chin-Lin and Ramdya, Pavan},
editor = {VijayRaghavan, K},
volume = 11,
year = 2022,
month = {oct},
pub_date = {2022-10-26},
pages = {e81527},
citation = {eLife 2022;11:e81527},
doi = {10.7554/eLife.81527},
url = {https://doi.org/10.7554/eLife.81527},
keywords = {population imaging, two-photon microscopy, descending neuron, walking, grooming, limb},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```
