# Delay-Embedding-based-Forecast-Machine
- The source code of paper "Multi-step-ahead Prediction from Short-term Data by Delay-Embedding-based Forecast Machine".
- This project is used to make time series forecasting on the target variable in a high-dimensional dynamical system only with short-term observed data.


## Data avalability
For the reason that all the data files are too large, thus all datasets can be download from [Google Drive](https://drive.google.com/file/d/1zzyf55_NkSlETgmuqeVFXR4rBoS1OpYm/view?usp=sharing). After downloading the zip file, you should extract all dataset folders in the zip file to the target folder `logs/data/`

## Environment requirements

- python = 3.6
- tenforflow = 2.1
- cuda-version = 10.1
- cudnn-version = 7.6.5

We suggest that you run the code with Pycharm.

## Training and making predicitons

- Firstly, you should modify the variable `DATA_BASE_DIR`(in `line 10, data/data_processing.py`) to the full path of the data folder in your computer.

- We release the sample training codes and predicting codes corresponding to the Lorenz time invariant dataset, Lorenz 96 system, and KS equation, which are located at folder `forecast/lorenz_time_invariant/`, `forecast/lorenz96/`, and `forecast/ks/`, respectively. The script `train.py` is used for training and the script `eval.py` is used for evaluation after training the model. 

- We can make predictions on other datasets by modify the given sample codes on Lorenz time invariant dataset.

## Experiment reuslts

(a)

![The prediciton results of Traffic dataset.](./demo_gif/traffic.gif)

The dynamical changes of the predicted and real traffic speeds in these six locations. 

(b)

![The prediciton results of Typhoon route dataset.](./demo_gif/typhoon.gif)

The predicted and real moving routes of typhoon Marcus starting from 23:00:00 3/20/2018 to 22:00:00 3/21/2018.

## Citation
Please cite this paper if you find the repository helpful:

    @article{peng2024defm,
      title={DEFM: Delay-embedding-based forecast machine for time series forecasting by spatiotemporal information transformation},
      author={Peng, Hao and Wang, Wei and Chen, Pei and Liu, Rui},
      journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
      volume={34},
      number={4},
      year={2024},
      publisher={AIP Publishing}
    }
