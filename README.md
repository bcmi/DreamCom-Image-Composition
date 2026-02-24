# DreamCom-Image-Composition


This is the official repository for the following research paper:

> **DreamCom: Finetuning Text-guided Inpainting Model for Image Composition**  [[arXiv]](https://arxiv.org/pdf/2309.15508.pdf)<br>
>
> Lingxiao Lu, Bo Zhang, Li Niu <br>

We divide generative composition methods into token-to-object methods and object-to-object methods (e.g., [ObjectStitch](https://github.com/bcmi/ObjectStitch-Image-Composition), [ControlCom](https://github.com/bcmi/ControlCom-Image-Composition)). DreamCom belongs to token-to-object methods. Currently, token-to-object methods have not shown clear advantage over object-to-object methods. 

## Online Demo

Try this [online demo](http://libcom.ustcnewly.com/) for image composition (object insertion) built upon [libcom](https://github.com/bcmi/libcom) toolbox and have fun!

[![]](https://github.com/user-attachments/assets/87416ec5-2461-42cb-9f2d-5030b1e1b5ec)

## Task

Given a few (3~5) reference images for a subject, we aim to customize a generative model, which can insert this subject into an arbitrary background image. In the obtained composite image, the foreground subject should be compatible with the background in terms of illumination, geometry, and semantics.

Technically, we finetune a text-guided inpainting model based on the reference images of one subject, during which a special token is associated with this subject. Then, we apply the finetuned model to a new background image. 

<p align='center'>  
  <img src='./figures/task.jpg'  width=60% />
</p>

## Our MureCom Dataset

Our MureCom dataset can be downloaded from [[Dropbox]](https://www.dropbox.com/scl/fi/o0y2r3685hakcm2eiwubz/MureCom.zip?rlkey=xmy24607rs1ejdlnrusnok7rk&st=vs4fs7hh&dl=0) or [[Baidu Cloud]](https://pan.baidu.com/s/1of-NO5QzJ8GQNmxABzWjzg?pwd=qg8r). Note that MureCom is extended from our previous [FOSCom](https://github.com/bcmi/ControlCom-Image-Composition/tree/main?tab=readme-ov-file#foscom-dataset) dataset. This folder consists of 32 category subfolders, where each subfolder contains the following data:

- **Backgrounds**: Each subfolder includes 20 background images suitable for that category. These background images are stored in the `bg` folder together with their bounding boxes to insert the foreground object.
- **Foregrounds**: Each subfolder includes 3 sets of foreground images, in which each set contains 5 images for the same foreground object. Three sets of foreground images are stored in the `fg1`, `fg2`, and `fg3` folders together with their masks.


## Code and Model

Our code is based on the basic code from the [diffusers library](https://github.com/huggingface/diffusers). Also, our model's format follows the rules set by the diffusers framework.

1.  Dependencies

    *   Python == 3.8
    *   Pytorch == 1.11.0
    *   Run

        ```bash
         conda env create -f environment.yml
        ```
2.  Download Models

    Please download the files and models from https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main, and put them in the `diffusers-v1-5-inpaint/` folder.
    
4.  Dataset

    You can download our MureCom Dataset, and put it under the folder `MureCom/` for training and testing.
5.  Train

    You can execute the following code snippet to initiate the training process for a model. This model is specifically designed to establish a correlation between the unique train located in the `MureCom/Train/fg1` directory and the concept represented by `"a photo of sks train"`. The trained model will be subsequently stored within the `models/` directory.

    ```bash
    accelerate launch --num_processes=1 train.py --fg_name fg1 --image_num 5 --package_name="Train" --class_name="train" --pretrained_model_name_or_path="diffusers-v1-5-inpaint"  --instance_data_dir="MureCom/" --output_dir="models/" --instance_prompt="a photo of sks " --background_prompt="background" --resolution=512 --train_batch_size=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=400 --gradient_accumulation_steps=1 
    ```
6.  Test

    You can execute the code provided below to utilize our recently trained model for generating sks train against the backgrounds found in the `MureCom/Train/bg` directory. The generated results will be appropriately stored in the corresponding location within the `MureCom/Train/result/fg1/` directory.

    ```bash
    python test.py --background_prompt="background" --do_crop --package_name="Train" --fg_name fg1 --image_num 5 --class_name="train"
    ```
7.  Train and test for all instances in MureCom

    ```bash
    bash train_and_test.sh
    ```

## Experiments

We show our results compared with some baselines. 

<p align='center'>  
  <img src='./figures/result.png'  width=90% />
</p>



## Other Resources

+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive on image composition: [the 3rd edition](https://arxiv.org/abs/2106.14490)
