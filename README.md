# DreamCom-Image-Composition


This is the official repository for the following research paper:

> **DreamCom: Finetuning Text-guided Inpainting Model for Image Composition**  [[arXiv]](https://arxiv.org/pdf/2309.15508.pdf)<br>
>
> Lingxiao Lu, Bo Zhang, Li Niu <br>
>

## Task

Given a few (3~5) reference images for a subject, we aim to customize a generative model, which can insert this subject into an arbitrary background image. In the obtained composite image, the foreground subject should be compatible with the background in terms of illumination, geometry, and semantics.

Technically, we finetune a text-guided inpainting model based on the reference images of one subject, during which a special token is associated with this subject. Then, we apply the finetuned model to a new background image. 

<p align='center'>  
  <img src='./figures/task.jpg'  width=60% />
</p>

## Code and Model

This work is still in progress. We will release code and model when they are ready.

Currently, we finetune the text-guided inpainting model from https://huggingface.co/runwayml/stable-diffusion-inpainting. However, the performance of this model is unsatisfactory. A better text-guided inpainting model is in high demand. 


## Experiments

We show our results compared with some baselines. 

<p align='center'>  
  <img src='./figures/result.png'  width=60% />
</p>

## Our MureCom Dataset


Our MureCom dataset is available in the `/MureCom` folder. This folder consists of 32 subject subfolders, where each subfolder contains the following data:

- **Backgrounds**: Each subfolder includes 20 background images corresponding to that subject. These background images are stored in the `bg` folder together with their bounding boxes.
- **Foregrounds**: Each subfolder includes 3 sets of foreground images, with each set containing 5 images. The foreground images are stored in the `fg1`, `fg2`, and `fg3` folders together with their masks.



## Other Resources

+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive on image composition: [the 3rd edition](https://arxiv.org/abs/2106.14490)
