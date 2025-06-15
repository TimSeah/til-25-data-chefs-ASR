# TIL-25 Data Chefs - ASR Challenge

**Hackathon:** TIL-25 Hackathon
**Team:** Data Chefs
**Author:** dgxy2002

## 📖 Description

This repository contains the solution for the ASR (Automatic Speech Recognition) challenge as part of the TIL-25 Hackathon. The primary goal was to train an effective ASR model.

*(You can add more specific details about the challenge problem here if you like.)*

## 💻 Technologies Used

*   **Python:** Core programming language for model development and scripting.
*   **Jupyter Notebook:** Used for experimentation, data exploration, and model training iterations.
*   **Shell Scripts:** For automation of tasks like data preprocessing, training initiation, etc.
*   **Whisper:** Utilized as the ASR model for this challenge.
*   **NumPy, Pandas, Matplotlib, PyTorch/TensorFlow** (Specify which of these were actually used, or add/remove as appropriate)

## ⚙️ Working Process & Solution

This section outlines the general steps taken to address the ASR challenge.

### 1. Data Collection & Preparation
*   **Dataset Used:** (Describe the dataset(s) used, e.g., public datasets like LibriSpeech, Common Voice, or custom collected data. Mention size, type of audio, language, etc.)
*   **Preprocessing:** (Detail the steps taken to clean and prepare the audio data for the Whisper model, e.g., resampling, noise reduction, silence trimming, format conversion, augmentation.)
*   **Labeling:** (If custom data was used, how was it transcribed? E.g., manual transcription, tools used.)

### 2. Model Selection & Architecture
*   **Model Choice:** The **Whisper** model by OpenAI was chosen for this ASR challenge. (Explain why Whisper was selected, e.g., its strong performance on diverse datasets, availability of pre-trained versions, ease of use.)
*   **Architecture Details:** Whisper is a transformer-based encoder-decoder model. (You can add more details if you used a specific variant or made modifications, but generally, this is a good starting point.)
*   **Pre-trained Models:** (Specify which version of the pre-trained Whisper model was used, e.g., tiny, base, small, medium, large. Mention if it was fine-tuned.)

### 3. Training Process
*   **Environment Setup:** (Briefly mention the environment, e.g., local machine specs with GPU, cloud VM (AWS, GCP, Azure), specific Python/library versions like `openai-whisper`.)
*   **Training Configuration:** (If fine-tuning Whisper: Key hyperparameters, loss functions, optimizers, batch size, number of epochs. If using Whisper out-of-the-box for transcription, this section might be more about inference parameters.)
*   **Fine-tuning:** (If a pre-trained Whisper model was fine-tuned, describe the fine-tuning strategy, dataset used for fine-tuning, and any specific techniques applied.)
*   **Challenges Faced:** (Any significant challenges during data preparation, training/fine-tuning, or inference with Whisper and how they were overcome.)

### 4. Evaluation
*   **Metrics Used:** (How was the model performance measured? E.g., Word Error Rate (WER) is standard for ASR. Character Error Rate (CER) can also be used.)
*   **Validation Strategy:** (How was the model validated? E.g., using a dedicated validation set. Describe the characteristics of this set.)
*   **Test Set Performance:** (Results on the final test set using the chosen Whisper model.)

### 5. Results & Key Findings
*   **Final Model Performance:** (Summarize the best WER/CER achieved with Whisper.)
*   **Insights:** (Any interesting insights gained from using Whisper, its performance on specific types of audio, or comparison to other approaches if applicable.)
*   **Visualizations:** (Consider linking to or embedding examples of ASR output if possible, or tables summarizing performance.)

## 🚀 Setup and Usage

### Prerequisites
*   Python version 3.10+
*   Git & Git LFS
*   CUDA version 12.2.1+

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/lolkabash/til-25-data-chefs-ASR.git
    cd til-25-data-chefs-ASR
    ```
2.  (If Git LFS was used for model files, etc.)
    ```bash
    git lfs pull
    ```
3.  Install dependencies:
    ```bash
    # conda env create -f environment.yml
    # conda activate your_env_name
    ```

### Running the Code
*   **Data Preparation:**
    ```bash
    # e.g., python scripts/prepare_data.py
    ```
*   **Training:**
    *(Explain how to run the training scripts or notebooks.)*
    ```bash
    # e.g., python train_ASR.py --config configs/my_config.yaml
    # or jupyter notebook PaddleASR_Training/CreateLabel.ipynb (based on previous interactions)
    ```
*   **Inference/Prediction:**
    *(Explain how to use the trained model for predictions.)*
    ```bash
    # e.g., python predict.py --image_path path/to/image.png --model_path path/to/model
    ```

## 📁 File Structure
```
til-25-data-chefs-ASR/
├── PaddleASR_Training/         # Main training scripts, notebooks, and model files (as per previous interactions)
│   ├── CreateLabel.ipynb
│   └── pretrained_models/
│       └── en_PP-ASRv4_rec_train/
│           ├── best_accuracy.pdparams
│           └── best_accuracy.pdopt
├── configs/                    # Configuration files for training
├── data/                       # Placeholder for datasets (ensure .gitignore if data is large and not in LFS)
├── notebooks/                  # Jupyter notebooks for exploration, analysis
├── scripts/                    # Utility scripts (data preprocessing, evaluation)
├── src/                        # Source code for model, utilities
├── .gitattributes              # For Git LFS tracking
├── .gitignore
└── README.md
```

## 🙏 Acknowledgements
*   Mention any datasets, pre-trained models, or codebases that were particularly helpful.
*   Thank you to my Data Chef teammates: Darren, Freddie, and Felix, for whom without this challenge would not have been possible.
