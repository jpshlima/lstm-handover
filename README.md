# Welcome to 'Deep Learning-Based Handover Prediction for 5G and Beyond Networks' repository

Although the 5G New Radio standard empowers the mobile communication networks with diverse technologies such as Massive MIMO, mmWave deployments, and much more, some network functionalities still do not explore the potential of assembling Artificial Intelligence to their methodologies. The handover procedure is planned very similarly to in older 3GPP networks, based on simple power measurement comparisons and rudimentary parameter tuning, such as Time-To-Trigger and Hysteresis. This work develops and evaluates with simulations and real network data a new Deep Learning approach to support the handover triggering decision toward a data-driven procedure for next-generation networks. Our solution relies on predicting future samples of standard Reference Signals using Long Short-Term Memory Networks (LSTM) in the first stage. After, the predicted power samples are sent to a binary classification algorithm to identify if the time series will lead or not to a handover triggering. The results show a mean absolute error of around 0.6 dB predicting power signal samples and over 97% of accuracy, indicating the future handover trigger moment. Finally, we discuss possible use cases to implement our model, including Open RAN and MEC architectures.

## Scripts features

Scripts with 'lstm_rsrp.py' are the ones responsible for creating the LSTM model to predict future RSRP samples.
The 'data_processing' scripts concatenate many different data files, create the final dataframe to serve as models' inputs.
The 'sampling_and_classify.py' files apply the selected class-balancing techniques (SMOTE and Tomek Links) and also implement the classifiers, with training and cross-validation steps.
The data is also found here, allowing reproducibility.

## Citation

If you use our scripts and dataset, please cite our ICC paper. Here is a suitable BibTeX entry:

```python
@inproceedings{joao_2023, 
title={Deep Learning-Based Handover Prediction for 5G and Beyond Networks}, 
DOI={TBD}, 
booktitle={IEEE International Conference on Communications (ICC)}, 
author={João P. S. H. Lima and Alvaro A. M. de Medeiros, Eduardo P. de Aguiar, Edelberto F. Silva, Vicente A. de Sousa Jr., Marcelo L. Nunes and Alysson L. Reis}, 
year={2023}, 
pages={TBD} 
}
```

## Authors
- João Paulo S. H. Lima
- Alvaro A. M. de Medeiros
- Eduardo P. de Aguiar
- Edelberto F. Silva
- Vicente A. de Sousa Jr.
- Marcelo L. Nunes
- Alysson L. Reis

## Acknowledgments
- This study was financed in part by FUNTTEL/Finep and the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001;
- The proof of concept simulations provided by this Letter was supported by High Performance Computing Center (NPAD/UFRN).
