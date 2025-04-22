# Uni3C
[Preprint] Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation

[[arXiv]](https://arxiv.org/abs/2504.14899) [[Project Page]](https://ewrfcas.github.io/Uni3C/)

### Abstract

Camera and human motion controls have been extensively studied for video generation, but existing approaches typically address them separately, suffering from limited data with high-quality annotations for both aspects.
To overcome this, we present **Uni3C**, a unified 3D-enhanced framework for precise control of both camera and human motion in video generation. Uni3C includes two key contributions. 
First, we propose a plug-and-play control module trained with a frozen video generative backbone, PCDController, which utilizes unprojected point clouds from monocular depth to achieve accurate camera control. 
By leveraging the strong 3D priors of point clouds and the powerful capacities of video foundational models, PCDController shows impressive generalization, performing well regardless of whether the inference backbone is frozen or fine-tuned. 
This flexibility enables different modules of Uni3C to be trained in specific domains, i.e., either camera control or human motion control, reducing the dependency on jointly annotated data.
Second, we propose a jointly aligned 3D world guidance for the inference phase that seamlessly integrates both scenic point clouds and SMPL-X characters to unify the control signals for camera and human motion, respectively.
Extensive experiments confirm that PCDController enjoys strong robustness in driving camera motion for fine-tuned backbones of video generation. 
Uni3C substantially outperforms competitors in both camera controllability and human motion quality. Additionally, we collect tailored validation sets featuring challenging camera movements and human actions to validate the effectiveness of our method.