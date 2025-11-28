## 1. Why should we incorporate frequency domain information?

Anomalies that can only be found by frequency domain

Pattern-wise Anomalies

Frequency changes

## 2. what kind of frequency domain transformation should we use

Paper : [Beyond the Time Domain: Recent Advances on Frequency Transforms in Time Series Analysis](https://arxiv.org/html/2504.07099v1)

fourier transformation (for comparsion)

wavelet transformation, suitable for non-stationary time series data.
can extract local information.

## 3. What should we know about wavelet transformation?

real value/ complex value wavelet transform

### 3.1. filters (mother wavelet)

Haar, Mexican Hat, Morlet

### 3.2 Invertibility

$$
C_{\psi} = \int_{0}^{\infty} \frac{|\Psi(\omega)|^2}{\omega} d\omega < \infty
$$


### WST

Wavelet scatter transformation

### DWT, CWT, DCT

All kinds of frequency domain transformations

## 4. How can we use the information gained from wavelet transformation ?

Frequency domain autoencoder
Use the invertibility of the transformation. The admissibility condition of the mother wavelet

Can also take the matrices it generates as the input for isolation forest

## 5. Does it actually improve the performance for anomaly detection?

[Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection](https://dl.acm.org/doi/10.1145/3589334.3645556)

[Wavelets for fault diagnosis of rotary machines: A review with applications](https://www.researchgate.net/publication/259098512_Wavelets_for_fault_diagnosis_of_rotary_machines_A_review_with_applications)

Find papers or findout during the experiments.，

## 6. How to find suitable time series data for benchmarking?

Numenta Anomaly Benchmark (NAB)

The NAB dataset is one of the most popular and well-designed benchmarks for time-series pattern anomalies.

Yahoo S5 Benchmark
Created by Yahoo Labs, this dataset suite is another crucial resource for collective anomaly detection, particularly in system monitoring.

## 7. Run experiments

Probably still use delos since my account is still available.

Will make a comparison with Fourier Transform Wavelet Transform and a baseline


Guten Tag dataset



### 7.1 Combine with DEAN model?

## 8. Paper writing

What template
The scale of the contents
number of pages