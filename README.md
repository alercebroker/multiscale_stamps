# Multiscale Stamps for Real-time Classification of Alert Streams

Public repository for "Multiscale Stamps for Real-time Classification of Alert Streams" by Reyes-Jainaga et al. 2023, published in The Astrophysical Journal Letters.

DOI 10.3847/2041-8213/ace77e

## /src

This folder contains the code used to train the classifiers. The data is not stored in the repository. This files can be useful to check how the models are implemented in code.

## /utilities

Here are some Jupyter notebooks that create multiscale stamps from FITS files. One of them creates and displays a multiscale stamp from ZTF data, the data source of the paper. The other one starts from a DECAM observation, which is useful because it has a pixel size similar to the Vera C. Rubin Observatory.

The WCS-related code in the DECAM notebook might have errors, as I'm not an expert on the topic (coordinate systems and projections).
