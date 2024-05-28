# Improving Accuracy-robustness Trade-off via Pixel Reweighted Adversarial Training
Official PyTorch implementation of the ICML 2024 paper: Improving Accuracy-robustness Trade-off via Pixel Reweighted Adversarial Training (link will be uploaded later).

#### Abstract
Adversarial training (AT) trains models using adversarial examples (AEs), which are natural images modified with specific perturbations to mislead the model.
These perturbations are constrained by a predefined perturbation budget $\epsilon$ and are equally applied to each pixel within an image. 
However, in this paper, we discover that not all pixels contribute equally to the accuracy on AEs (i.e., robustness) and accuracy on natural images (i.e., accuracy). 
Motivated by this finding, we propose Pixel-reweighted AdveRsarial Training (PART), a new framework that partially reduces $\epsilon$ for less influential pixels, guiding the model to focus more on key regions that affect its outputs.
Specifically, we first use class activation mapping (CAM) methods to identify important pixel regions, then we keep the perturbation budget for these regions while lowering it for the remaining regions when generating AEs. 
In the end, we use these pixel-reweighted AEs to train a model.
PART achieves a notable improvement in accuracy without compromising robustness on CIFAR-10, SVHN and TinyImagenet-200, justifying the necessity to allocate distinct weights to different pixel regions in robust classification. 

#### Figure 1: The proof-of-concept experiment.
![motivation](https://github.com/JiachengZ01/PART/blob/main/images/motivation.jpg)
We find that fundamental discrepancies exist among different pixel regions. Specifically, we segment each image into four equal-sized regions (i.e., ul, short for upper left; ur, short for upper right; br, short for bottom right; bl, short for bottom left) and adversarially train two ResNet-18 on CIFAR-10 using standard AT with the same experiment settings except for the allocation of $\epsilon$. The robustness is evaluated by $\ell_{\infty}$-norm PGD-20. With the same overall perturbation budgets (i.e., allocate one of the regions to $6/255$ and others to $12/255$), we find that both natural accuracy and adversarial robustness change significantly if the regional allocation on $\epsilon$ is different. For example, by changing $\epsilon_{\rm{br}} = 6/255$ to $\epsilon_{\rm{ul}} = 6/255$, accuracy gains a 1.23\% improvement and robustness gains a 0.94\% improvement.

#### Figure 2: The illustration of our method.
![pipeline](/PART/tree/main/images/pipeline.jpg)
Compared to AT, PART leverages the power of CAM methods to identify important pixel regions. Based on the class activation map, we element-wisely multiply a mask to the perturbation to keep the perturbation budget $\epsilon$ for important pixel regions while shrinking it to $\epsilon^{\rm low}$ for their counterparts during the generation process of AEs.

### Requirement
- This codebase is written for ```python3``` and ```pytorch```.
- To install necessay python packages, run ```pip install -r requirements.txt```.

### Data
- Please download and place the dataset into the 'data' directory.

### Run Experiments
#### Train and Evaluate PART
```
python3 train_eval_part.py
```
#### Train and Evaluate PART-T
```
python3 train_eval_part_t.py
```

#### Train and Evaluate PART-M
```
python3 train_eval_part_m.py
```

### License and Contributing
- This README is formatted based on [the NeurIPS guideline](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post any issues via Github.
