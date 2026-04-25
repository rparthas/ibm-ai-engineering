# IBM AI Engineering Projects

This repository contains notebooks and projects completed as part of the IBM AI Engineering learning path. The work spans classical machine learning, PyTorch fundamentals, deep learning, Keras and PyTorch model comparisons, vision transformers, and transformer-based NLP.

## Project Directory Overview

| Directory | Focus | Contents |
|---|---|---|
| `2.unsupervised-keras/` | Unsupervised learning with Keras | Diffusion model implementation lab |
| `4.1.linear-pytorch/` | Linear regression with PyTorch | Introductory PyTorch linear modeling notebooks |
| `4.2.logistic-pytorch/` | Logistic regression and softmax classification with PyTorch | Binary and multiclass classification notebooks |
| `5.dl-pytorch/` | Deep learning with PyTorch | Neural networks, activation functions, dropout, batch normalization, initialization, momentum, CNNs, and MNIST/FashionMNIST labs |
| `6.1.final-proj/` | Final applied machine learning projects | Wine classification, League of Legends match prediction, and fruit image classification |
| `6.capstone/` | Keras/PyTorch classifier comparison and vision transformer capstone | Keras classifier, PyTorch classifier, comparative analysis, ViT labs, and CNN-ViT integration evaluation |
| `9.transformers/` | Transformer models | Self-attention, positional encoding, classification transformers, and decoder causal language models |

## Notebooks and Projects

### 2. Unsupervised Keras

- `2.unsupervised-keras/M04_Lab_Implementing_Diffusion_Models.ipynb`  
  Implements and evaluates a diffusion model workflow using Keras.

### 4.1. Linear PyTorch

- `4.1.linear-pytorch/4-3.ipynb`  
  PyTorch linear modeling lab.
- `4.1.linear-pytorch/4-4.ipynb`  
  Additional PyTorch linear modeling lab.

### 4.2. Logistic PyTorch

- `4.2.logistic-pytorch/5.ipynb`  
  Logistic regression with PyTorch.
- `4.2.logistic-pytorch/5-1.ipynb`  
  Logistic classification lab.
- `4.2.logistic-pytorch/5-2.ipynb`  
  Additional logistic classification lab.
- `4.2.logistic-pytorch/Softmax Classifier.ipynb`  
  Multiclass classification with a softmax classifier.
- `4.2.logistic-pytorch/softmax classifier-2.ipynb`  
  Additional softmax classification notebook.

### 5. Deep Learning with PyTorch

- `5.dl-pytorch/Activation Functions.ipynb`
- `5.dl-pytorch/Activation and Maxpooling.ipynb`
- `5.dl-pytorch/Batch Normalization.ipynb`
- `5.dl-pytorch/CNN Small Images.ipynb`
- `5.dl-pytorch/CNN Smple Example.ipynb`
- `5.dl-pytorch/CNN With batch normalization.ipynb`
- `5.dl-pytorch/Convolutional Neural Network for Anime Image Classification-v1.ipynb`
- `5.dl-pytorch/Deep Neural Networks.ipynb`
- `5.dl-pytorch/Deeper Neural Networks.ipynb`
- `5.dl-pytorch/Dropout Classification.ipynb`
- `5.dl-pytorch/Dropout Regression.ipynb`
- `5.dl-pytorch/Even More Neurons.ipynb`
- `5.dl-pytorch/FashionMNISTProject-v1.ipynb`
- `5.dl-pytorch/Initialization with weights.ipynb`
- `5.dl-pytorch/Momentum with different polynomials.ipynb`
- `5.dl-pytorch/Momentum.ipynb`
- `5.dl-pytorch/MoreNeurons.ipynb`
- `5.dl-pytorch/MultipleIO.ipynb`
- `5.dl-pytorch/Neural networks with one hidden layer.ipynb`
- `5.dl-pytorch/OneLayer.ipynb`
- `5.dl-pytorch/Test Activation on MNIST.ipynb`
- `5.dl-pytorch/Test Initialization With Activation.ipynb`
- `5.dl-pytorch/Test initalization.ipynb`
- `5.dl-pytorch/convolution.ipynb`

These notebooks cover core deep learning topics including fully connected neural networks, activation functions, initialization, optimization with momentum, dropout, batch normalization, max pooling, convolutional neural networks, and image classification.

### 6.1. Final Projects

- `6.1.final-proj/Deep Neural Network for Wine Classification-v1.ipynb`  
  Applies a deep neural network to wine classification.
- `6.1.final-proj/Final Project League of Legends Match Predictor-v2.ipynb`  
  Builds a model to predict League of Legends match outcomes.
- `6.1.final-proj/TEST-PROJ-v1.ipynb`  
  Fruit image classification project using transfer learning.
- `6.1.final-proj/project_overview.md`  
  Overview document for the final project work.

### 6. Capstone

- `6.capstone/Lab_M2L1_Train_and_Evaluate_a_Keras-Based_Classifier.ipynb`  
  Trains and evaluates a Keras-based classifier.
- `6.capstone/Lab_M2L2_Implement_and_Test_a_PyTorch-Based_Classifier.ipynb`  
  Implements and tests a PyTorch-based classifier.
- `6.capstone/Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb`  
  Compares Keras and PyTorch model implementations.
- `6.capstone/Lab_M3L1_Vision_Transformers_in_Keras.ipynb`  
  Vision transformer implementation in Keras.
- `6.capstone/Lab_M3L2_Vision_Transformers_in_PyTorch.ipynb`  
  Vision transformer implementation in PyTorch.
- `6.capstone/lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb`  
  Integrates and evaluates CNN and ViT approaches for land classification.

### 9. Transformers

- `9.transformers/Self-Attention and Positional Encoding.ipynb`  
  Explores self-attention and positional encoding concepts.
- `9.transformers/M3-L2-Applying Transformers for Classification-v2.ipynb`  
  Applies transformer models to classification tasks.
- `9.transformers/Decoder Causal Language Models.ipynb`  
  Works with decoder-only causal language models.

## Dependencies

Dependencies are managed in `pyproject.toml`. The project includes commonly used AI and data science libraries such as:

- `jupyterlab`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `torch`
- `torchvision`
- `torchaudio`
- `torchtext`
- `torchdata`
- `transformers`
- `seaborn`
- `plotly`
- `dash`

## Usage

Install the project dependencies with your preferred Python package manager, then launch JupyterLab from the repository root:

```bash
uv sync
uv run jupyter-lab
```

Alternatively, open JupyterLab directly and navigate to the notebook directory you want to run.

## Notes

Large generated artifacts such as model checkpoint files are intentionally excluded from version control. Notebook outputs may reflect prior runs and can vary depending on package versions, random seeds, and available hardware.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
