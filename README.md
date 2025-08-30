# IBM AI Engineering Projects

This repository contains two machine learning projects developed as part of the IBM AI Engineering Professional Certificate.

## Projects

### 1. Fruit Classification

This project focuses on building a fruit image classifier using transfer learning. 

- **Notebook:** `final-proj/TEST-PROJ-v1.ipynb`
- **Dataset:** [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits)
- **Model:** VGG16 (pre-trained on ImageNet)
- **Description:** The project demonstrates how to fine-tune a pre-trained model on a custom dataset of fruit images to classify fruits effectively. It covers data preparation, model building, training, evaluation, and visualization of results.

### 2. Diffusion Model Implementation

This project is a lab on implementing, training, and evaluating diffusion models using Keras.

- **Notebook:** `unsupervised-keras/M04_Lab_Implementing_Diffusion_Models.ipynb`
- **Dataset:** MNIST
- **Description:** The lab provides a practical understanding of diffusion model architectures, data processing, model training, and performance evaluation. It walks through the process of building a simple diffusion model to denoise MNIST images.

## Dependencies

The projects use the following Python libraries:

- `jupyterlab`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `tensorflow`

You can install the dependencies using the `pyproject.toml` file with a package manager like `pip` or `uv`.

## Usage

To run the projects, you need to have JupyterLab installed. You can then navigate to the respective project directories and open the Jupyter notebooks.

```bash
# Navigate to the final project directory
cd final-proj
# Start JupyterLab
jupyter-lab
```

```bash
# Navigate to the unsupervised keras directory
cd unsupervised-keras
# Start JupyterLab
jupyter-lab
```

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
