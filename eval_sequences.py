import os
import subprocess
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datetime import datetime

class ImageSequenceEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.results = []
        
    def combine_image_sequence(self, image_files, output_path):
        try:
            cmd = ["convert", "+append"] + image_files + [output_path]
            subprocess.run(cmd, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error combining images: {e}")
            return None

    def process_directory(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        sequences = {}
        for file in sorted(input_path.glob("*.jpg")):
            sequence_name = file.stem.split('_')[0]
            if sequence_name not in sequences:
                sequences[sequence_name] = []
            sequences[sequence_name].append(str(file))

        for sequence_name, image_files in sequences.items():
            output_file = output_path / f"{sequence_name}_combined.jpg"
            combined_image = self.combine_image_sequence(image_files, str(output_file))
            
            if combined_image:
                result = self.evaluate_sequence(combined_image, sequence_name)
                self.results.append(result)

    def evaluate_sequence(self, image_path, sequence_name):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}",
                    },
                    {
                        "type": "text",
                        "text": "Describe what is happening in this image sequence, from a video scene. "
                               "Describe it as a single action. If the person is doing a trick, try to guess which one."
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "sequence_name": sequence_name,
            "image_path": str(image_path),
            "description": output_text,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process and evaluate image sequences')
    parser.add_argument('input_dir', help='Directory containing input image sequences')
    parser.add_argument('output_dir', help='Directory for combined images and results')
    parser.add_argument('--results-file', default='evaluation_results.json',
                       help='Output JSON file for evaluation results')
    
    args = parser.parse_args()
    
    evaluator = ImageSequenceEvaluator()
    evaluator.process_directory(args.input_dir, args.output_dir)
    evaluator.save_results(args.results_file)
    
    for result in evaluator.results:
        print(f"\nSequence: {result['sequence_name']}")
        print(f"Description: {result['description']}")
        print("-" * 80)

if __name__ == "__main__":
    main()