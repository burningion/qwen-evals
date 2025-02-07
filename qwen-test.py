from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
#)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://output.jpg",
            },
            {"type": "text", "text": "Describe what is happening in this image sequence, from a video scene. Describe it as a single action. If the person is doing a trick, try to guess which one. As an example for what your output should be: A man is walking down a sidewalk, with his hands in his pockets. There's green grass around him, and there are cars in a street behind him."},
        ],
    }
]



'''
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file://leo1.jpg",
                    "file://leo2.jpg",
                    "file://leo3.jpg",
                ],
            },
            {"type": "text", "text": "Describe what is happening in this video scene"},
        ],
    }
]

'''

# output for this: The video appears to be shot from the perspective of a vehicle's rearview mirror, capturing a scene outside. The view shows a person riding a skateboard on a paved surface, likely a parking lot or a similar area. The person is wearing a cap and seems to be moving at a moderate pace. The surroundings include some trees and a few buildings in the background, suggesting an urban or suburban setting. The lighting indicates it might be daytime.
# (bad!)

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://alex-output.jpg",
            },
            {"type": "text", "text": "Describe what is happening in this image sequence, from a video scene. Describe it as a single action. If the person is doing a trick, try to guess which one. As an example for what your output should be: A man is walking down a sidewalk, with his hands in his pockets. There's green grass around him, and there are cars in a street behind him."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

