# Hyperspectral Image Segmentation
A notebook of hyperspectral image segmentation as part of the "Pixxel Open Data Challenge" . 

Note: Please view the .ipynb notebooks to visualize the results obtained.

## Data
The data used here collected by NASA's Program EO-1 which was launched on November 2000, to capture high resolution images of the earth surface using hyperion hyper-spectral imager. There are 220 unique spectral channels collected with a complete
spectrum covering from 357 - 2576 nm. The Level 1 Radiometric product
has a total of 242 bands but only 198 bands are calibrated. 

Two tiles were used for two separate analysis.

Tiles:
1. EO1H1430452010208110Kt_1GST
2. EO1H1480472016328110PZ_1GST

## Libraries Used

* PIL
* tifffile
* numpy
* matplotlib
* skicit-learn
* rasterio

## Methodology

The TIF Images were initially imported into an array through a function from the TIFFILE library. Principle Component Anlaysis was used for dimensionality reduction after which, the K-Means clustering algorithm was used as the unsupervised classification algorithm which segmented the three most important principle components. Individual bands from the first three principle components were merged to obtain the final false color image.

## Results


The unsupervised segmentation (K-Means) was measured using the Davies-Bouldin score and it came out to be 0.460 for Tile-1 and 0.5852 for Tile-2



## References
* https://stackoverflow.com/
* https://www.kaggle.com
* https://archive.usgs.gov/archive/sites/eo1.usgs.gov/EO1userguidev2pt320030715UC.pdf
* https://www.researchgate.net/publication/319277363_Hyperspectral_image_classification_A_k-means_clustering_based_approach
* https://www.researchgate.net/publication/281037741_Hyperspectral_hyperion_imagery_analysis_and_its_application_using_spectral_analysis
* https://www.researchgate.net/publication/221912997_Survey_of_Multispectral_Image_Fusion_Techniques_in_Remote_Sensing_Applications
* https://www.researchgate.net/publication/333448474_A_SURVEY_OF_HYPERSPECTRAL_IMAGE_SEGMENTATION_TECHNIQUES_FOR_MULTIBAND_REDUCTION
