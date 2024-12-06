
# FedMDH: A Federated Learning Framework for Effective Sharing of Multi-Dimensional Heterogeneous Materials Data

This Repository contains the code that accompanies the paper [FedMDH: A Federated Learning Framework for Effective Sharing of Multi-Dimensional Heterogeneous Materials Data], including the source codes for the project's platform foundation ([MatSwarm](https://rdcu.be/dYm1R)) and FedMDH related algorithms ([FedMDH](fabric-mge-backend/apps/fl/FedMDH)）.

Supplementary Materials and Videos could also be found [here](fabric-mge-backend/apps/fl/FedMDH). For further details, please refer to the [Slides](FedMDH_Slides.pdf).

In the field of materials science, due to various factors such as material sources, testing equipment, and technical methods, the data distributions across different organizations are often non-identical and non-independent (non-i.i.d.) . This data heterogeneity can manifest in various forms, including 1) feature space disparity, 2) sample imbalance, and 3) label distribution variance. We define it as multi-dimensional heterogeneity (MDH). To overcome these challenges, we introduce FedMDH, a federated learning framework designed to tackle Multi-Dimensional Heterogeneity. While FedMDH is applicable to various downstream tasks, this work focuses on the widespread, complex, and underexplored regression tasks in materials science.<br> 
<br>
Our experiments on real-world datasets from the [NMDMS platform](http://mged.nmdms.ustb.edu.cn/analytics/) demonstrate that FedMDH significantly outperforms existing methods, offering superior accuracy and enhanced generalization across multi-dimensional heterogeneity. The successful deployment of FedMDH within the NMDMS platform further unlocks the full potential of material data, accelerates material discovery, and meets the demands of high-throughput computing and experimentation.

# Project basis: MatSwarm
MatSwarm is the official demonstration project for the NMDMS federated computing system. FedMDH is tested based on the MatSwarm platform. 

FedMDH Showcase is displayed in the following video. More introductory videos could be referred to [here](https://www.youtube.com/playlist?list=PLO3YbKD6ntr95rEowaeu9kEG_NHyUXJ4l).

[![FedMDH Showcase](https://res.cloudinary.com/marcomontalbano/image/upload/v1730950248/video_to_markdown/images/youtube--f7w9tbDljNQ-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/f7w9tbDljNQ "FedMDH Showcase")

Experiments in papers and presentation videos can be replicated based on the platform. Please refer to the installation and instructions via [MatSwarm Instructions](MatSwarm-README.en.md). More details could be referred to the following paper: [MatSwarm](https://rdcu.be/dYm1R).
