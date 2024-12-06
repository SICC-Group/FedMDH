
# FedMDH: A Federated Learning Framework for Effective Sharing of Multi-Dimensional Heterogeneous Materials Data

This Repository contains the code that accompanies the paper [FedMDH: A Federated Learning Framework for Effective Sharing of Multi-Dimensional Heterogeneous Materials Data], including the source codes for the project's platform foundation ([MatSwarm](https://rdcu.be/dYm1R)) and FedMDH related algorithms ([FedMDH](fabric-mge-backend/apps/fl/FedMDH)）.

Supplementary Materials and Videos could also be found [here](fabric-mge-backend/apps/fl/FedMDH). For further details, please refer to the [Slides](FedMDH_Slides.pdf).

In the field of materials science, due to various factors such as material sources, testing equipment, and technical methods, the data distributions across different organizations are often non-identical and non-independent (non-i.i.d.) . This data heterogeneity can manifest in various forms, including 1) feature space disparity, 2) sample imbalance, and 3) label distribution variance. We define it as multi-dimensional heterogeneity (MDH). To overcome these challenges, we introduce FedMDH, a federated learning framework designed to tackle Multi-Dimensional Heterogeneity. While FedMDH is applicable to various downstream tasks, this work focuses on the widespread, complex, and underexplored regression tasks in materials science.<br> 
<br>
Our experiments on real-world datasets from the [NMDMS platform](http://mged.nmdms.ustb.edu.cn/analytics/) demonstrate that FedMDH significantly outperforms existing methods, offering superior accuracy and enhanced generalization across multi-dimensional heterogeneity. The successful deployment of FedMDH within the NMDMS platform further unlocks the full potential of material data, accelerates material discovery, and meets the demands of high-throughput computing and experimentation.

# Project basis: MatSwarm
MatSwarm is the official demonstration project for the NMDMS federated computing system. FedMDH is tested based on the MatSwarm platform. 

FedMDH Showcase is desplayed in the following vedio. More demonstrate videos could be referred to [here](https://www.youtube.com/playlist?list=PLO3YbKD6ntr95rEowaeu9kEG_NHyUXJ4l)：

[![FedMDH Showcase](https://res.cloudinary.com/marcomontalbano/image/upload/v1730950248/video_to_markdown/images/youtube--f7w9tbDljNQ-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/f7w9tbDljNQ "FedMDH Showcase")

Experiments in papers and presentation videos can be replicated based on the platform. Please refer to the installation and instructions via this link:[MatSwarm Instructions](MatSwarm-README.en.md)

More details could be referred to the following paper: [MatSwarm](https://rdcu.be/dYm1R).

```
@article{wang2024matswarm,
  title={Matswarm: trusted swarm transfer learning driven materials computation for secure big data sharing},
  author={Wang, Ran and Xu, Cheng and Zhang, Shuhao and Ye, Fangwen and Tang, Yusen and Tang, Sisui and Zhang, Hangning and Du, Wendi and Zhang, Xiaotong},
  journal={Nature Communications},
  volume={15},
  number={9290},
  pages={1-14},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## FedMDH Dataset

**Perovskite Dataset**

**Training Dataset 1:** 1231 samples，9 features were derived through sampling, label distribution (−2.81,0.128)<br>
**Training Dataset 2:** 891 samples，10 features were derived through sampling, label distribution (-2.03,0.027)<br>
**Training Dataset 3:** 1066 samples，7 features were derived through sampling, label distribution (-0.91,0.138)<br>
**Test Dataset:** 322 samples，12 features were derived through sampling, label distribution (-1.85,0.905)

## Feature Engineering

In our experiments, we used the MatMiner tool to extract various aspects of material characteristics through the following four feature engineering methods:

**(1) StrtoComposition:**
```
from matminer.featurizers.conversions import StrToComposition
dfmatw = StrToComposition().featurize_dataframe(df, "formula")
```
**(2) ElementProperty:**
```
from matminer.featurizers.composition import ElementProperty 
ep_feat = ElementProperty.from_preset(preset_name="magpie")
dfmatw = ep_feat.featurize_dataframe(dfmatw, col_id="composition") 
```
**(3) CompositionToOxidComposition:**
```
from matminer.featurizers.conversions import CompositionToOxidComposition
dfmatw = CompositionToOxidComposition().featurize_dataframe(dfmatw, "composition",ignore_errors=True)
```
**(4) OxidationStates:**
```
from matminer.featurizers.composition import OxidationStates
os_feat = OxidationStates()
dfmatw = os_feat.featurize_dataframe(dfmatw, "composition_oxid",ignore_errors=True)
```
## Environment Setup and Dependencies

To ensure compatibility with Python 3.9.19, follow these setup instructions:

1. **Set Up Python 3.9.19**: Ensure Python 3.9.19 is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/release/python-3919/) or install it via a package manager.

2. **Create a Virtual Environment**: It is recommended to create a virtual environment to manage dependencies and avoid conflicts.

   ```bash
   python3.9 -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install Required Packages**: Once the virtual environment is activated, install the dependencies. To run this program, ensure you have the following dependencies installed. You can install these packages via `pip` or by setting up a virtual environment for easy dependency management.

**Core Libraries and Modules**

```python
Package             Version
------------------- -----------
colorama            0.4.6      
contourpy           1.2.1      
cycler              0.12.1     
et-xmlfile          1.1.0      
filelock            3.15.4     
fonttools           4.53.1     
fsspec              2024.6.1   
h5py                3.11.0     
importlib_resources 6.4.3      
Jinja2              3.1.4      
joblib              1.4.2
kiwisolver          1.4.5
MarkupSafe          2.1.5
matplotlib          3.9.2
mpmath              1.3.0
networkx            3.2.1
numpy               1.26.4
openpyxl            3.1.5
packaging           24.1
pandas              2.2.2
pillow              10.4.0
pip                 24.2
pyparsing           3.1.2
python-dateutil     2.9.0.post0
pytz                2024.1
scikit-learn        1.5.1
seaborn             0.13.2
setuptools          72.1.0
six                 1.16.0
sympy               1.13.2
threadpoolctl       3.5.0
torch               2.4.0
torchaudio          2.4.0
torchvision         0.19.0
tqdm                4.66.5
typing_extensions   4.12.2
tzdata              2024.1
wheel               0.43.0
zipp                3.20.0
```

**Custom Modules**

1). **FLAlgorithms**  
   - Users: `UserFedMDH`, `User`, `RegressionTracker`
   - Servers: `Server`, `FedMDH`
   - Training: `RKLDivLoss`, `FedProxOptimizer`
   
2). **Utilities**  
   - Model utilities: `create_model`, `convert_data`, `read_data`, `read_user_data`, `aggregate_user_data`, `create_generative_model`, `create_discriminator_model`
   - Plot utilities: `plot_utils`
   
3). **Het_Update**  
   - `Het_LocalUpdate`, `het_test_img_local_all`, `train_preproc`, `aggregate_models`, `wass_loss`
   
4). **Het_Nets**  
   - `get_reg_model`, `get_preproc_model`
   
5). **Prior Regularization**  
   - `Prior`

---

You can install the primary dependencies using `pip` with the following command:

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn
```

Additional dependencies and custom modules should be included or linked to this project’s repository. 

Ensure that any remaining module-specific dependencies are properly configured within your environment.


## Run Experiments: 

All source codes and data of FedMDH are in the folder [FedMDH](fabric-mge-backend/apps/fl/FedMDH).

Refer to the main file "main.py" which allows running all experiments.

#### Run experiments on the *Perovskite Dataset* Dataset:
```
python main.py --dataset Material-alpha0.1-ratio0.5 --algorithm FedMDH --batch_size 64 --num_glob_iters 200 --local_epochs 10 --num_users 3

----

=======
# FedMDH
