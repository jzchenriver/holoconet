# [GRSL] Holistic Modularization of Local Contrast in the End-to-End Network for Infrared Small Target Detection

Single-frame infrared small target detection is a challenging task due to the noise and clutter interference. Recent emerging deep learning methods achieve superior detection performance compared to traditional model-driven methods. However, these data-driven methods do not possess the explicit gradient encoding capability of local contrast methods. To overcome the restriction, we propose a holistic local contrast network (HoLoCoNet) in this letter to gradually couple the local contrast into the end-to-end network, which consists of a multiscaled multidirectional attention module (M2AM) to directly processes the input image, a multibranch dilated difference convolution module (D2CM) for secondary refinement of the multiscale features extracted by the backbone network, and a semantic-enhanced aggregation module (SEAM) for bottom–up feature fusion by enhancing shallow features with deep semantic knowledge. The experimental results on the widely accepted NUDT-SIRST and IRSTD-1K dataset demonstrate the rationality and effectiveness of the proposed HoLoCoNet with the probability of detection reaching 99.2 and 94.3. The source codes are available at http://github.com/jzchenriver/HoLoCoNet.

https://ieeexplore.ieee.org/document/10268945

[Chen G, Wang Z, Wang W, et al. Holistic Modularization of Local Contrast in the End-to-end Network for Infrared Small Target Detection[J]. IEEE Geoscience and Remote Sensing Letters, 2023.]

The test codes and trained models of HoLoCoNet on the NUDT-SIRST and IRSTD-1k datasets.

For validating the metrics in the paper, 

1. Downloading the dataset.

2. Putting them in ./datasets/[name of the dataset]/images/ and ./datasets/[name of the dataset]/masks/, respectively.

3. Run test.py

If you find the work helpful, please give me a star and cite the paper.

Thank you!
