export MODEL_NAME="diffusers-v1-5-inpaint"
export INSTANCE_DIR="MureCom/"
export OUTPUT_DIR="models/"


list1=("Person" "Coffee_cup" "Goose" "Horse" "Guitar" "Fish" "Laptop" "Monkey" "Motorcycle" "Pen" "Taxi" "Train" "Watch" "Duck" "Dog" "Couch" "Cat" "Car" "Camera" "Cake" "Airplane" "Bird" "Book" "Box" "Bottle" "Bread" "Bus" "Computer_keyboard" "Mobile_phone" "Picture_frame" "Waste_container" "Toilet")
list2=("person" "coffee cup" "goose" "horse" "guitar" "fish" "laptop" "monkey" "motorcycle" "pen" "taxi" "train" "watch" "duck" "dog" "couch" "cat" "car" "camera" "cake" "airplane" "bird" "book" "box" "bottle" "bread" "bus" "computer keyboard" "mobile phone" "picture frame" "waste container" "toilet")

# list1=("Train")
# list2=("train")


length=${#list1[@]}

for ((i=0; i<$length; i++)); do
    package_name=${list1[i]}

    class_name=${list2[i]}
    
    accelerate launch --num_processes=1 train.py --fg_name fg1 --image_num 5 --package_name=$package_name --class_name="$class_name" --pretrained_model_name_or_path=$MODEL_NAME  --instance_data_dir=$INSTANCE_DIR --output_dir=$OUTPUT_DIR --instance_prompt="a photo of sks " --background_prompt="background" --resolution=512 --train_batch_size=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=400 --output_medium --gradient_accumulation_steps=1 #--gradient_checkpointing --mixed_precision=fp16
    python test.py --background_prompt="background" --do_crop --package_name=$package_name --fg_name fg1 --image_num 5 --class_name="$class_name"
    
done

