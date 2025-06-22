# SPEECH-RECOGNITION-SYSTEM

*COMPANY* : CODTECH IT SOLUTION

*NAME* : TALARI BHAVANA

*INTERN ID* : CT04DF313

*DOMAIN* : Artificial Intelligence

*DURATION* : 4 weeks 

*MENTOR* : NEELA SANTOSH

*DISCRIPTION* : The provided Python code builds a **speech-to-text system** using **Wav2Vec 2.0**, a powerful pre-trained model from Facebook AI, which performs automatic speech recognition (ASR). This system takes a `.wav` audio file as input and converts the spoken content into written text. Several modern machine learning and audio processing tools are integrated to accomplish this, including **PyTorch**, **Torchaudio**, and the **Transformers library** from Hugging Face.

At the core of the system lies the **Wav2Vec2.0 model**, specifically `"facebook/wav2vec2-base-960h"`, which is a pre-trained speech recognition model trained on 960 hours of Librispeech data. This model is hosted on **Hugging Face's Transformers Hub**, a repository of thousands of ready-to-use models for tasks such as NLP, computer vision, and audio processing. The model is loaded using the `Wav2Vec2ForCTC` class, which implements Connectionist Temporal Classification (CTC) — an architecture designed to map sequences of audio features to text when timing alignment is unknown. It works hand-in-hand with `Wav2Vec2Processor`, which handles feature extraction, tokenization, and decoding of the audio input.

To handle the audio data, the system uses **Torchaudio**, which is part of the PyTorch ecosystem and provides tools for loading, transforming, and processing audio signals. In this code, `torchaudio.load()` is used to read the waveform and sample rate from a `.wav` file. Since Wav2Vec2 expects audio sampled at **16,000 Hz**, a check is included to resample the audio using `torchaudio.transforms.Resample` if the original sample rate is different.

The audio waveform is then preprocessed by the `Wav2Vec2Processor`, which extracts input features and prepares them in a format suitable for the model. These inputs are passed into the model for inference. The model outputs a sequence of **logits**, which are unnormalized probabilities for each character or token in the vocabulary. These logits are converted to predicted IDs using `torch.argmax()`, and finally decoded into human-readable text using the processor’s `.decode()` function.

The use of **PyTorch** is essential in this workflow, as both the Wav2Vec2 model and Torchaudio are built upon it. PyTorch provides a dynamic computation graph and high-performance tensor operations which are crucial for model inference. `torch.no_grad()` is used during prediction to disable gradient tracking, which speeds up computation and reduces memory usage since we are not training the model.

This system showcases how pre-trained models can be seamlessly integrated into real-world applications using minimal code. It demonstrates the practical power of combining AI libraries such as **Transformers for deep learning**, **Torchaudio for audio I/O and processing**, and **PyTorch for backend tensor computations**. By abstracting much of the complexity, these tools allow developers, researchers, and students to focus on innovation and implementation, rather than model training from scratch. This speech-to-text tool can be extended for voice assistants, transcription software, language learning applications, and more.
