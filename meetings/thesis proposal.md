Hi Xuanhao,  
  
here is the paper I promised to you for online learning:  
[https://arxiv.org/abs/2506.13217](https://arxiv.org/abs/2506.13217)  
  
and this is my paper about constant learning that I thought you could  
maybe add wavelets to:  
[https://arxiv.org/abs/2504.20733](https://arxiv.org/abs/2504.20733)  
  




1. anomalies that can only be found by frequency domain
answer：



2. what kind of frequency domain transformation should we use 
Fourier transform and wavelet transform, why do we want wavelet transform?
which kernel from wavelet should we choose?

如果我们把这个积分看作一次卷积或内积运算，那么每个频率 $\omega$  对应一个「滤镜」：

$hω(t)=eiωth_\omega(t) = e^{i\omega t}hω​(t)=eiωt$


于是：

> 傅里叶变换就是用无数个不同频率的正弦波去“扫描”信号，看它对每个频率的响应强度。

这和小波做法几乎一样，只不过小波基底在时间上有限，而傅里叶基底在整个时间轴上无限延展

The difference between wavelet transform and the fourier transform.

wavelet transform tell you not only about what frequency are there but also about what time the appears.

different kind of filters:
Haar
Mexican Hat
Morlet

The accuracy of wavelet transformation


|影响因素|含义|对结果的影响|
|---|---|---|
|尺度步长 Δa|尺度取值间隔是否足够细|控制频率分辨率|
|平移步长 Δb|时间扫描密度|控制时间分辨率|
|小波类型 ψ(t)|母小波形状是否适合信号特征|决定匹配灵敏度|
|信号采样率|原始数据是否足够密集|决定整体时域精确性|

如何理解傅立叶变换的值？

如何理解小波变换的值？
于是计算出的 W(a,b)W(a,b)W(a,b) 本身可能是：

- 实数 → 如果母小波 ψ 是实值函数；
- 复数 → 如果母小波 ψ 包含正弦振荡项（例如 Morlet 小波）。

有实值小波和复值小波

对anomaly detection 是否有帮助
### 实值小波：

- 只能反映“变化有多强”；
- 对突变、边缘敏感；
- 但无法区分相同强度下不同振动模式。

### 复值小波：

- 除了变化强度，还能捕捉“变化的方向与时间偏移”（相位信息）；
- 能区分“模式不同但能量相似”的情况；
- 对噪声和非平稳信号更加鲁棒。



3. how to incoperate frequencey domain information to the model?

### 1. Frequency-domain Autoencoder

将信号转换到 FFT 或 STFT 域后输入自编码器，通过重构误差判断是否异常。  
代表工作：**Zhou et al., IEEE Trans. Ind. Informatics, 2020**


4. does it actually improve the performance?


5. how to find time series anomaly detection benchmark?

ADBench 

6. Run experiments and compare those metrics








