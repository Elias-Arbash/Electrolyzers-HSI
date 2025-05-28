# Electrolyzers-HSI: Close-Range Multi-Scene  Hyperspectral Imaging Benchmark Dataset
[HZDR](https://hzdr.de) - [Hif_Exploration](https://www.iexplo.space/)
## Overview

Our primary focus is to enhance the non-invasive optical analysis of E-waste materials, including Electrolyzers in order to accelerate the recovery of critical raw materials in recycling mainstreams. We aim to develop a smart multisensor network that utilizes RGB cameras and hyperspectral imaging to improve material identification in real time.
![http://url/to/img.png](https://github.com/Elias-Arbash/Electrolyzers-HSI/blob/main/images/acquisition3.png)

## Research Paper

This GitHub repository corresponds to the research paper titled "Electrolyzers-HSI: Close-Range Multi-Scene  Hyperspectral Imaging Benchmark Dataset." The paper introduces RGB-Hyperspectral Imaging (HSI) benchmark segmentation dataset for Electrolyzer materials. You can access the paper [here](https://arxiv.org/abs/2505.20507).

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

For detailed code instructions, please refer to the code documentation. More information about the methodology and experiments can be found in the paper.

## Data Access

To utilize the dataset, download it from [this link](https://rodare.hzdr.de/record/3668).

## Contributions

All comments and contributions are welcome. The repository can be forked, edited, and pushed to different branches for enhancements. Feel free to contact me directly at e.arbash@hzdr.de or via our [HiF-Explo](https://www.iexplo.space/).

## License

The code is licensed under the a CC BY-NC-ND (Creative Commons Attribution Non-Commercial No Derivatives 4.0 International license. Any further development and application using this work should be opened and shared with the community.

## Acknowledgment 
The authors express their gratitude to EIT RawMaterials for funding the project 'RAMSES-4-CE' (KIC RM 19262) and BMBF for funding AI4H2 as part of the ReNaRe (3245129018-03HY111D) in the flagship cluster "H2Giga". Appreciation is extended to the European Regional Development Fund (EFRE) and the Land of Saxony for their support in funding the computational equipment under the project ’CirculAIre.’ Special thanks go to Yuleika Carolina Madriz Diaz and Filipa Simoes for their assistance in data acquisition.

## Citation

When using the materials of the work and the dataset, please cite them as follows:

**Latex:**

@misc{arbash2025electrolyzershsicloserangemultiscenehyperspectral,
      title={Electrolyzers-HSI: Close-Range Multi-Scene Hyperspectral Imaging Benchmark Dataset}, 
      author={Elias Arbash and Ahmed Jamal Afifi and Ymane Belahsen and Margret Fuchs and Pedram Ghamisi and Paul Scheunders and Richard Gloaguen},
      year={2025},
      eprint={2505.20507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20507}, 
}
