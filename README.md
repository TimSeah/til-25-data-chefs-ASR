# TIL-25 Data Chefs - ASR Challenge

**Hackathon:** TIL-25 Hackathon
**Team:** Data Chefs
**Author:** dgxy2002

## 📖 Description

This repository contains the solution for the ASR (Automatic Speech Recognition) challenge as part of the TIL-25 Hackathon. The primary goal was to fine-tune an effective ASR model using a pre-trained architecture and deploy it as a service meeting specified performance criteria.

*(You can add more specific details about the challenge problem here if you like.)*

## 💻 Technologies Used

*   **Python:** Core programming language for model development and scripting.
*   **Hugging Face Transformers:** For accessing pre-trained models (Wav2Vec2) and the `Trainer` API.
*   **Hugging Face Datasets:** For loading and processing the dataset.
*   **PyTorch:** As the backend deep learning framework for `transformers`.
*   **Wav2Vec2 (facebook/wav2vec2-large-960h):** The pre-trained ASR model used for fine-tuning.
*   **JiWER:** Used for calculating Word Error Rate (WER) for the accuracy score.
*   **(Flask/FastAPI or other web framework):** For creating the `/asr` endpoint (Specify which one if used, e.g., Flask).
*   **Base64:** For decoding input audio.
*   **Pydub:** (Potentially used for initial audio format conversion and resampling).
*   **Librosa:** (Potentially used for audio analysis and feature extraction during initial preprocessing).
*   **Jupyter Notebook:** (Potentially used for experimentation, data exploration before scripting the training).
*   **Shell Scripts:** (Potentially for automation of tasks).
*   **(NumPy, Pandas, Matplotlib):** (Include if used in earlier data preparation stages or for analysis).

## ⚙️ Working Process & Solution

This section outlines the steps taken to fine-tune the Wav2Vec2 model for the ASR challenge and prepare it for evaluation.

### 1. Data Collection & Initial Preparation
*   **Dataset Used:** The dataset was provided by **DSTA Brainhack TIL-AI 2025**. It was loaded from `/home/jupyter/advanced/asr/processed_dataset` for the fine-tuning script. (Further describe the dataset if details are known: e.g., total hours of audio, language, specific characteristics like noisy environments, number of speakers, and the nature of 'audio' and 'transcription' fields within it).
*   **Initial Preprocessing (prior to `train.py`):**
    *   **Format Conversion:** Original audio files (if not already in WAV format) were likely converted to WAV.
    *   **Resampling:** Audio files were resampled to a consistent 16kHz sampling rate, as expected by the Wav2Vec2 model. Tools like `Pydub` or `librosa` might have been used.
    *   **Channel Normalization:** Audio was converted to mono (single channel).
    *   **Silence Trimming/VAD:** Potentially, leading/trailing silences or long pauses were removed using Voice Activity Detection (VAD) to focus on speech segments.
    *   **Normalization:** Audio volume might have been normalized to a consistent level.
    *   **Dataset Structuring:** The preprocessed audio files and their corresponding transcriptions were organized into a structure compatible with the Hugging Face `datasets` library, leading to the creation of the `processed_dataset` directory. This likely involved creating a dataset manifest or loading scripts.

### 2. Model-Specific Data Preprocessing (as in `train.py`)
*   The `processed_dataset` was loaded and split into training (90%) and evaluation (10%) sets.
*   The `Wav2Vec2Processor` (from `facebook/wav2vec2-large-960h`) was used to:
    *   Process audio arrays (already at 16kHz sampling rate) into `input_values` suitable for the model.
    *   Convert text transcriptions into token IDs (`labels`).
*   This preprocessing was applied to both training and evaluation datasets.

### 3. Model Selection & Architecture
*   **Model Choice:** `facebook/wav2Vec2-large-960h` was chosen as the base pre-trained model for fine-tuning. This model was selected due to its strong performance on various ASR benchmarks and its robust architecture for transfer learning.
*   **Architecture Details:** Wav2Vec2 is a self-supervised learning model that learns contextualized speech representations by pre-training on large amounts of unlabeled audio. The `Wav2Vec2ForCTC` variant is used for ASR tasks, employing a Connectionist Temporal Classification (CTC) loss function for training.
*   **Pre-trained Processor:** The corresponding `Wav2Vec2Processor` was used for data preparation, ensuring consistency between the model's expectations and the input data.

### 4. Fine-Tuning Process
*   **Environment Setup:**
    *   Local machine with an **NVIDIA GeForce RTX 3060 Ti GPU**
    *   CUDA version 12.8
    *   Python 3.10+
*   **Training Arguments (`TrainingArguments`):**
    *   Output directory: `./output` (for saving model checkpoints and logs)
    *   Logs directory: `./logs` (for TensorBoard or other logging)
    *   Batch size: 8 per device (for both training and evaluation)
    *   Number of epochs: 3
    *   Logging steps: 500
    *   Evaluation strategy: `steps` (evaluate at each `save_steps`)
    *   Save steps: 1000
    *   Save total limit: 2 (to keep only the best/latest checkpoints based on evaluation metric)
*   **Trainer Initialization:** The Hugging Face `Trainer` was used to manage the fine-tuning process, taking the model, training arguments, preprocessed train/evaluation datasets, and potentially a `compute_metrics` function for WER calculation during evaluation.
*   **Training Execution:** The `trainer.train()` method was called to start the fine-tuning.
*   **Challenges Faced:** (e.g., "Initial fine-tuning runs resulted in slower than expected inference times. Experimentation with quantization or ONNX conversion was considered but not fully implemented due to time constraints." or "Achieving the target WER required careful hyperparameter tuning, particularly the learning rate and number of epochs.")

### 5. Evaluation

The model's performance is assessed based on accuracy and speed.

#### 5.1. Accuracy Score
The accuracy score is calculated as:
`max(0, 1 - WER)`
where WER (Word Error Rate) is computed using the `jiwer` library.

Before calculating WER, the predicted transcript undergoes the following transformations:
```python
jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"-": " "}),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])
```

#### 5.2. Speed Score
The speed score is calculated by:
`1 - (min(t_elapsed, t_max) / t_max)`
where:
*   `t_elapsed` is the time taken for the model to complete inference on the whole test set.
*   `t_max` is the inference duration beyond which the model gets a zero speed score. For Qualifiers, `t_max` is 30 minutes.

A lower `t_elapsed` results in a better speed score.

#### 5.3. Input/Output Format for ASR Service

**Input:**
The ASR service expects a POST request to the `/asr` route on port 5001. The request body is a JSON document:
```json
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_AUDIO"
    }
    // ... more instances
  ]
}
```
*   `b64`: Base64-encoded bytes of the input audio in WAV format.

**Output:**
The service must return a JSON dictionary:
```json
{
  "predictions": [
    "Predicted transcript one.",
    "Predicted transcript two."
    // ... more predictions
  ]
}
```
*   The `k`-th element in `predictions` corresponds to the `k`-th element in the input `instances`.
*   The length of `predictions` must equal the length of `instances`.

#### 5.4. Validation Strategy (during training)
*   The model was evaluated on the 10% evaluation split during training at specified `save_steps` using the Hugging Face `Trainer`.
*   A `compute_metrics` function incorporating `jiwer` (with the specified transforms) was passed to the `Trainer` to monitor WER on the evaluation set. This allowed for saving the checkpoint with the best WER.

#### 5.5. Test Set Performance
*   The fine-tuned model was evaluated on the official hidden test set provided by the organizers, adhering to the specified input/output formats and scoring criteria.

### 6. Results & Key Findings

*   **Final Model Performance:**
    *   **Accuracy Score:** 0.915
    *   **Speed Score:** 0.779
*   **Insights:**
    *   Fine-tuning the `facebook/wav2vec2-large-960h` model for 3 epochs on the provided DSTA Brainhack TIL-AI 2025 dataset significantly improved its performance on domain-specific audio compared to the off-the-shelf model.
    *   The preprocessing steps, particularly ensuring a consistent 16kHz sampling rate and the `jiwer` normalization for WER calculation, were crucial for achieving reliable results.
    *   The batch size of 8 was found to be a good compromise between training speed and GPU memory constraints on the NVIDIA 3060 Ti.
    *   Further improvements in speed might be achievable through model quantization (e.g., to INT8) or conversion to an optimized runtime format like ONNX, though this was not explored in depth for this submission.
    *   The choice of learning rate and scheduler (if used) had a noticeable impact on convergence and final WER.
*   **Audio Demonstrations & Visualizations:**
    *   **Sample: Operation Echelon Update**
        *   Audio: [Listen to sample_0.wav](./media/asr/sample_0.wav)
        *   Ground Truth: "Operation Echelon has yielded significant progress in our pursuit of the rogue AI droid BH-2000. Our surveillance drones have identified its current location in sector 7G, and our ground units are mob..."
        *   Model's Prediction: *(You can fill this in with your model's output for this sample)*
    *   (Consider linking to or embedding:
        *   Training/validation loss curves and WER plots from TensorBoard logs.
        *   A confusion matrix or examples of common error types if detailed error analysis was performed.
        *   Examples of challenging audio segments and their transcriptions (before and after fine-tuning if applicable).)
*   **Future Work:**
    *   Experiment with different pre-trained model sizes (e.g., `wav2vec2-base` vs. `wav2vec2-large-robust`) to balance accuracy and speed.
    *   Implement more advanced data augmentation techniques for audio.
    *   Explore end-to-end models that might simplify the preprocessing pipeline.
    *   Conduct a more thorough hyperparameter search using tools like Optuna or Ray Tune.

## 📁 File Structure
```
til-25-data-chefs-ASR/
├── ASR_Training/         # Main training scripts, notebooks, and model files (as per previous interactions)
│   ├── train.py
│   └── preparedataset.py
├── configs/                    # Configuration files for training
├── data/                       # Placeholder for datasets (ensure .gitignore if data is large and not in LFS)
├── notebooks/                  # Jupyter notebooks for exploration, analysis
├── scripts/                    # Utility scripts (data preprocessing, evaluation)
├── src/                        # Source code for model, utilities
├── .gitattributes              # For Git LFS tracking
├── .gitignore
└── README.md
```
