# [Hydrological droughts of 2017-2018 explained by the Bayesian reconstruction of GRACE(-FO) fields](https://doi.org/10.1029/2022WR031997)
[Shaoxing Mo](https://scholar.google.com/citations?user=b5m_q4sAAAAJ&hl=en&oi=ao), [Yulong Zhong](https://scholar.google.com/citations?user=A7D2Kv4AAAAJ&hl=en), [Ehsan Forootan](https://scholar.google.com/citations?user=Yaor7_UAAAAJ&hl=en), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), [Wei Feng](https://scholar.google.com/citations?user=B5oOckcAAAAJ&hl=en), Xin Yin, Jichun Wu

This is a PyTorch implementation of Bayesian Convolutional Neural Network (BCNN) for reconstructing the missing GRACE(-FO) TWSA fields of 2017-2018 from hydroclimatic predictors in an image-to-image (field-to-field) regression manner. This rep is revised after [the rep](https://github.com/cics-nd/cnn-surrogate) by [Dr. Yinhao Zhu](https://scholar.google.com/citations?user=89uRjBkAAAAJ&hl=en).

## Dependencies
* python 3
* PyTorch
* h5py
* matplotlib
* scipy

## Network Training
```
python train_svgd.py
```

## BCNN network architecture

![BCNN architecture](https://github.com/njujinchun/BCNN4GRACE/blob/main/imgs/BCNN_arch-1.png)


## Reference GRACE(-FO) TWSA field, BCNN' reconstruction and predictive uncertainty

![](https://github.com/njujinchun/BCNN4GRACE/blob/main/imgs/BCNN_preditions.gif)

### The BCNN-identified drought regions during the GRACE and GRACE-FO gap (July 2017-May 2018)

![The BCNN-identified drought regions during the GRACE and GRACE-FO gap (July 2017-May 2018)](https://github.com/njujinchun/BCNN4GRACE/blob/main/imgs/BCNN_WSDI_201707_201805.png)



## Citation
See [Mo et al. (2022a)](https://doi.org/10.1029/2022WR031997) and [Mo et al. (2022b)](https://www.sciencedirect.com/science/article/pii/S0022169421012944) for more information. If you find this repo useful for your research, please consider to cite:

```
* Mo, S., Zhong, Y., Forootan, E., Shi, X., Feng, W., Yin, X., Wu, J. (2022a). Hydrological droughts of 2017–2018 explained by the 
Bayesian reconstruction of GRACE(-FO) fields. Water Resources Research, 58, e2022WR031997. https://doi.org/10.1029/2022WR031997

* Mo, S., Zhong, Y., Forootan, E., Mehrnegar, N., Yin, X., Wu, J., Feng, W., Shi, X. (2022b). Bayesian convolutional neural networks 
for predicting the terrestrial water storage anomalies during GRACE and GRACE-FO gap. Journal of Hydrology, 604, 127244. 
https://doi.org/10.1016/j.jhydrol.2021.127244
```
Related article: [Zhu, Y., & Zabaras, N. (2018). Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification. J. Comput. Phys., 366, 415-447.](https://www.sciencedirect.com/science/article/pii/S0021999118302341)

## Questions
Contact Shaoxing Mo (smo@nju.edu.cn) with questions or comments.
