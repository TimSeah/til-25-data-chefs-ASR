import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from datasets import load_from_disk
import torch

# Load the pre-trained model and processor (Using Wav2Vec2 as an example)
model_name = "facebook/wav2vec2-large-960h"  # You can replace this with any suitable ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load the processed dataset
def load_dataset():
    dataset = load_from_disk("/home/jupyter/advanced/asr/processed_dataset")
    return dataset

# Preprocess the dataset for ASR training
def preprocess_data(dataset):
    def process_example(example):
        # Apply processor to audio and transcriptions
        audio_input = processor(example['audio'], sampling_rate=16000, return_tensors="pt").input_values
        transcription = example['transcription']
        return {
            'input_values': audio_input[0],
            'labels': processor.text_to_ids(transcription)
        }

    return dataset.map(process_example)

# Prepare the training arguments
def get_training_arguments():
    return TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        do_train=True,
        do_eval=True,
    )

# Initialize the Trainer
def initialize_trainer(model, train_dataset, eval_dataset):
    training_args = get_training_arguments()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer

# Main training loop
if __name__ == "__main__":
    print("Loading the dataset...")
    
    # Load the processed dataset
    dataset = load_dataset()

    # Split the dataset into train and eval
    dataset = dataset.train_test_split(test_size=0.1)

    # Preprocess the data for ASR training
    train_dataset = preprocess_data(dataset['train'])
    eval_dataset = preprocess_data(dataset['test'])
    
    print("Dataset prepared and preprocessed.")
    
    # Initialize the trainer
    trainer = initialize_trainer(model, train_dataset, eval_dataset)
    
    print("Training started...")
    
    # Start training the model
    trainer.train()

    print("Training complete.")
