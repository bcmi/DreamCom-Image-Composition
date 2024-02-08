import PIL
import requests
import torch
from io import BytesIO
import numpy as np
import torchvision.transforms.functional as tf
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor
import random

device = "cuda"
random.seed(666)
import torch
torch.manual_seed(666)



def mask_bboxregion_coordinate(mask):
    w, h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask == 255)  # [length,2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:, 0])
        x_right = np.max(valid_index[:, 0])
        y_bottom = np.min(valid_index[:, 1])
        y_top = np.max(valid_index[:, 1])

    return x_left, x_right, y_bottom, y_top



def output_medium(iter,pipeline,package_name,class_name,image_num=5,fg_name='fg',background_prompt=None,do_crop=True):

    pipeline = pipeline.to(device)

    package_name =package_name
    class_name = class_name
    init_images = []
    for ij in range(20):
        image = Image.open(
            'testset2/' + package_name + '/bg/' + str(ij) + '.jpg')
        init_images.append(image)
        print(image.size)

    file_path ='MureCom/' + package_name + '/bbox.txt'
    bboxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[0:]:
            line = line.strip().split(' ')
            class_label = line[0]
            cc = 1
            while 'jpg' not in line[cc]:
                class_label = class_label + '_' + line[cc]
                cc += 1
            image_name = line[cc]
            x1, y1, x2, y2 = [int(coord) for coord in line[cc + 1:]]
            bboxes.append([x1,y1,x2,y2])

    save_path = 'MureCom/'+package_name+'/result/'+fg_name+'/ours_bgcroptest'+str(args.image_num)
    import os
    os.makedirs(save_path, exist_ok=True)


    for jj, bbox in enumerate(bboxes):

        x_left, y_bottom, x_right, y_top = bbox[0], bbox[1], bbox[2], bbox[3]
        W, H = (np.array(init_images[jj])).shape[:2]
        mask_array_new = np.zeros((W, H))
        y_top = min(y_top, W - 1)
        x_right = min(x_right, H - 1)

        new_temp = np.ones((y_top - y_bottom + 1, x_right - x_left + 1)) * 255
        mask_array_new[y_bottom:y_top + 1, x_left:x_right + 1] = new_temp
        mask_array_new = np.uint8(mask_array_new)
        mask_image = Image.fromarray(mask_array_new)

        mask_W=y_top - y_bottom + 1
        mask_H=x_right - x_left + 1

        prompt = "a photo of sks " + class_name

        if do_crop:

            if mask_W*mask_H*4<W*H:

                scale=0.5
                
            
                y_bottom=max(0,y_bottom-int(scale*mask_W))
                x_left=max(0,x_left-int(scale*mask_H))
                y_top=min(W-1,y_top+int(scale*mask_W))
                x_right=min(H-1,x_right+int(scale*mask_H))

                new_W=y_top-y_bottom+1
                new_H=x_right-x_left+1

                np_image=np.array(init_images[jj])
                np_mask=np.array(mask_image)
                
                np_image_big=np_image[y_bottom:y_top + 1, x_left:x_right + 1,:]
                np_mask_big=np_mask[y_bottom:y_top + 1, x_left:x_right + 1]

                mask_image = Image.fromarray(np_mask_big)
                init_image= Image.fromarray(np_image_big)




                init_image=init_image.resize((512,512),resample=Image.Resampling.LANCZOS)
                mask_image=mask_image.resize((512,512),resample=Image.Resampling.NEAREST)

                for ii in range(5):
                    image = pipeline(prompt=prompt,image=init_image, mask_image=mask_image,strength=1.0).images[0]

                    image=image.resize((new_H,new_W),resample=Image.Resampling.LANCZOS)
                    np_origin=np.array(init_images[jj])
                    np_image=np.array(image)   
                    np_origin[y_bottom:y_top + 1, x_left:x_right + 1,:]=np_image
                    image= Image.fromarray(np_origin)

                    image.save(save_path + '/iter_' + str(iter) + '_bbox_' + str(jj) + '_num_' +str(ii)+ '.jpg')
        
            else:
            
                

                init_image=init_images[jj].resize((512,512),resample=Image.Resampling.LANCZOS)
                mask_image=mask_image.resize((512,512),resample=Image.Resampling.NEAREST)
     
                for ii in range(5):
                    image = pipeline(prompt=prompt,image=init_image, mask_image=mask_image,strength=1.0).images[0]
                    image.save(save_path + '/iter_' + str(iter) + '_bbox_' + str(jj) + '_num_' +str(ii)+ '.jpg')
        else:


            init_image=init_images[jj].resize((512,512),resample=Image.Resampling.LANCZOS)
            mask_image=mask_image.resize((512,512),resample=Image.Resampling.NEAREST)

            for ii in range(5):
                image = pipeline(prompt=prompt,image=init_image, mask_image=mask_image,strength=1.0).images[0]
                image.save(save_path + '/iter_' + str(iter) + '_bbox_' + str(jj) + '_num_' +str(ii)+ '.jpg')




import argparse
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--class_name",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--package_name",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--image_num",
    type=int,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--fg_name",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--background_prompt",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--do_crop",
    default=False,
    action="store_true",
    help="",
)

args = parser.parse_args()
package_name=args.package_name
class_name=args.class_name
pathh='models/'+args.fg_name+str(args.image_num)+''

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    pathh,
                    torch_dtype=torch.float16,
                    requires_safety_checker=False,
                    safety_checker=None)

output_medium(iter=0,pipeline=pipeline,package_name=package_name,class_name=class_name,image_num=args.image_num,fg_name=args.fg_name,background_prompt=args.background_prompt,do_crop=args.do_crop)

