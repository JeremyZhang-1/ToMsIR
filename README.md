# ToMsIR: Task-oriented Multi-scene Image Restoration
# ToMsIR: Task-oriented Multi-scene Image Restoration for Visual Perception in Autonomous Vehicles
#  Introduction
ToMsIR is a task-oriented multi-scene image restoration framework designed for intelligent vehicle perception under adverse weather. It targets both single and mixed degradations, including haze, rain, snow, and their combinations, which often cause severe visibility reduction, detail loss, and complex nonlinear artifacts that degrade downstream perception reliability.

# Abstract
Adverse environmental conditions such as haze, rain, and snow, particularly when they co-occur, can severely degrade image quality and undermine the reliability of vision-based systems in intelligent vehicles. These degradations obscure structural details, reduce visibility, and introduce complex nonlinear interactions that challenge key perception tasks such as object detection, lane recognition, and scene understanding. To address these challenges, this paper presents a task-oriented multi-scene image restoration framework (termed ToMsIR) tailored for diverse single and mixed degradations. ToMsIR adopts a modular architecture that integrates multi-scale feature extraction, frequency-domain enhancement, and degradation-guided adaptive processing. In particular, a degradation-aware visual prompting mechanism is introduced, where the degradation classification network identifies the underlying degradation type and distribution to generate prompt-like representations that guide adaptive refinement. The shared-parameter encoder captures both local structures and global context, while the frequency enhancement module restores degradation-sensitive cues in the frequency domain. Furthermore, the adaptive fusion module dynamically adjusts restoration strategies based on the predicted degradation prompts, achieving fine-grained adaptation across complex scenes. Extensive experiments demonstrate that ToMsIR surpasses existing methods under haze, rain, snow, and their hybrid combinations. The restored images also yield more reliable inputs for downstream perception tasks in autonomous driving, highlighting the framework’s potential to enhance the robustness and safety of intelligent vehicles operating in adverse weather conditions.



# Prerequisites
```
conda create -n dehaze python=3.7
conda activate dehaze
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
python3 -m pip install scipy==1.7.3
python3 -m pip install opencv-python==4.4.0.46
```

# Result
Qualitative comparison of state-of-the-art dehazing methods on the RESIDE dataset. The figure presents the hazy input and the corresponding restored results produced by DCP, MSCNN, AODNet, FFANet, DehazeFormer, AiOENet, AirNet, TransW, MIRNet, WeatherDiff, MvkSR, DGdehazing, SFMN and ToMsIR, along with the ground truth reference.
![Figure_07](https://github.com/user-attachments/assets/8d3c5660-688a-416c-bfbf-a3f3107ad4c7)

Qualitative comparison of state-of-the-art deraining methods on the Rain100L dataset. The figure presents the rainy input and the corresponding restored results produced by DDN, DID, LPNet, DIG, DualGCN, AiOENet, AirNet, TransW, MIRNet, WeatherDiff, MvkSR, PADUM, CSUD and ToMsIR, along with the ground truth reference.
![Figure_08](https://github.com/user-attachments/assets/43802f04-bcab-4355-8735-dd2a621f1e6b)

Qualitative comparison of state-of-the-art desnowing methods on the CSD dataset. The figure presents the snowy input and the corresponding restored results produced by JSTASR, CodeNet, DRT, UMWT, FocalNet, AiOENet, AirNet, TransW, MIRNet, WeatherDiff, MvkSR, DSANet, SnowMaster and ToMsIR, along with the ground truth reference.
![Figure_09](https://github.com/user-attachments/assets/dcae155f-81b7-48df-9f09-5b04207fffdd)

Qualitative comparison of state-of-the-art multi-scene image restoration methods on the CDD-11 dataset. The figure presents the degraded input and the corresponding restored results produced by AirNet, TransW, MIRNet, WeatherDiff, MvkSR, USRNet and ToMsIR, along with the ground truth reference.
![Figure_10](https://github.com/user-attachments/assets/3a4bcae3-4370-4f26-9149-1ba7244974c1)


# License
The code and models in this repository are licensed under the WHUT License for academic and other non-commercial uses.
For commercial use of the code and models, separate commercial licensing is available. Please contact:
- Jingming Zhang (jeremy.zhang@whut.edu.cn)
- Yuxu LU (yuxulouis.lu@connect.polyu.hk)
