# Electrolyzers-HSI: Close-Range Multi-Scene  Hyperspectral Imaging Benchmark Dataset
[HZDR](https://hzdr.de) - [Hif_Exploration](https://www.iexplo.space/)
## Overview

Our primary focus is to enhance the non-invasive optical analysis of E-waste materials, including Electrolyzers in order to accelerate the recovery of critical raw materials in recycling mainstreams. We aim to develop a smart multisensor network that utilizes RGB cameras and hyperspectral imaging, to improve material identification in real time.
![http://url/to/img.png](https://github.com/Elias-Arbash/Electrolyzers-HSI/blob/main/images/acquisition3.png)

## Research Paper

This GitHub repository corresponds to the research paper titled "Electrolyzers-HSI: Close-Range Multi-Scene  Hyperspectral Imaging Benchmark Dataset." The paper introduces RGB-Hyperspectral Imaging (HSI) benchmark segmentation dataset for Electrolyzer materials. You can access the paper [Electrolyzers-HSI](https://arxiv.org/abs/2505.20507).

## Dataset Details

The dataset includes 55 scene of shredded Electrolyzer samples with RGB cameras and HSI data cubes containing 450 bands in the VNIR and SWIR range [400 - 2500]nm:
- 55 RGB images of shredded Electrolyzers scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 55 hyperspectral data cubes of those shredded Electrolyzers scanned with FENIX in the VNIR and SWIR ranges.
- 55 segmentation ground truth masks of five classes: Mesh - Steel_black - Steel_Grey - HTEL_Anode - HTEL_Cathode

<img src="https://github.com/Elias-Arbash/Electrolyzers-HSI/blob/main/images/data.png" width="50%">


## Code Repository

The repository includes the code for the inference pipeline, including data loading with utility functions. It provides the codes for the evaluated models and the majority voting using zero-shot predictions.

### Requirements

Install the libraries listed in the Requirements.txt file to use the code without errors. The codes require at least 1 GPU to run and handle the data.

### Usage

For detailed code instructions, please refer to the code documentation. More information about the methodology and experiments can be found in the paper [here](https://).

## Data Access

To utilize the dataset, download it from [this link](https://rodare.hzdr.de/record/3668).

## Contributions

All comments and contributions are welcome. The repository can be forked, edited, and pushed to different branches for enhancements. Feel free to contact me directly at e.arbash@hzdr.de or via our [website](https://www.iexplo.space/).

## License

The code is licensed under the Apache-2.0 license. Any further development and application using this work should be opened and shared with the community.

## Acknowledgment 
Appreciation is extended to the European Regional Development Fund (EFRE) and the Land of Saxony for their support
In funding the computational equipment under the project ’CirculAIre.’

## Citation

When using the materials of the work and the dataset, please cite them as follows:

**Word:**

To be provided soon !

**BibTeX:**

@misc{arbash2025electrolyzershsicloserangemultiscenehyperspectral,
      title={Electrolyzers-HSI: Close-Range Multi-Scene Hyperspectral Imaging Benchmark Dataset}, 
      author={Elias Arbash and Ahmed Jamal Afifi and Ymane Belahsen and Margret Fuchs and Pedram Ghamisi and Paul Scheunders and Richard Gloaguen},
      year={2025},
      eprint={2505.20507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20507}, 
}
