<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$dist(X_{S},X_{T})=\mu \lVert\frac{1}{n_{1}}\sum\limits_{i=1}^{n_{1}}\phi(x^{S}_{i})-\frac{1}{n_{2}}\sum\limits_{i=1}^{n_{2}}\phi(x^{T}_{i})\rVert +\\
 (1-\mu)\sum\limits_{c=1}^{C}\lVert\frac{1}{n_{c}}\sum\limits_{x^{S}_{i}\in \mathrm{X}_{S}^{(c)}}\phi(x^{S}_{i})-\frac{1}{m_{c}}\sum\limits_{x^{T}_{i}\in \mathrm{X}_{T}^{(c)}}\phi(x^{T}_{i})\rVert$$" style="border:none;">

# Balanced Distribution Adaptation

### INTRODUCTION
- A non-deep distribution adaptation method to decrease the MMD
distance of source and target domain for machine learning. 
----------------------------------------------
### MODIFICATION
- Different from original BDA,
when limited labeled data can be obtained on target
domain, they could be used to improve the performance.
----------------------------------------------
### RESULT
- accuracy on coil1: \
  ![Alt text](imgs/1534404869.jpg)
- feature visualization
----------------------------------------------
### REFERENCE
[1] Wang J, Chen Y, Hao S, et al. Balanced Distribution Adaptation for Transfer Learning[C]// IEEE International Conference on Data Mining. IEEE, 2017:1129-1134.