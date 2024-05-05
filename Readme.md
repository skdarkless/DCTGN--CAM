
# Sketch-supervised Histopathology Tumour Segmentation: Dual CNN-Transformer with Global Normalised CAM

Welcome to our GitHub repository dedicated to advancing histopathology tumour segmentation using a novel approach that combines Convolutional Neural Networks (CNNs) and Transformers with a Global Normalised Class Activation Map (CAM). This project aims to leverage sketch-supervised learning techniques to improve segmentation accuracy and interpretability in histopathological images.

## Repository Structure

```
.
├── keras_train.py          # Training script for Keras model
├── test.py                 # Script for model testing and evaluation
├── torch_train.py          # Training script for PyTorch model
└── utils/                  # Utility scripts and notebooks for CAM generation and patch cutting
    ├── cam1.ipynb
    ├── cam1_new.ipynb
    ├── cam1_torch.ipynb
    ├── cam2.ipynb
    ├── cam3.ipynb
    ├── cam4.ipynb
    ├── cam5.ipynb
    ├── cam6.ipynb
    ├── cam7.ipynb
    ├── cut_patch.ipynb
    └── cut_patch_testset.ipynb
```

## Features

- **Dual Model Architecture:** Integration of CNN and Transformer models to leverage both local and global contextual information.
- **Global Normalised CAM:** Improved method for generating class activation maps, enhancing the model's interpretability.
- **Sketch-supervised Learning:** Utilizes sketches as supervisory signals to refine segmentation in complex histopathology images.

## Getting Started

To get started with this project, clone the repository and install the required dependencies. Follow the instructions in each script and notebook for detailed steps on training and evaluating the models.

```bash
git clone https://github.com/skdarkless/DCTGN--CAM.git
cd DCTGN--CAM
pip install -r requirements.txt
```

## Dataset

This project is developed using a proprietary dataset of histopathological images. For access or more information on the dataset, please contact the authors.

## Updates

- **23-11**: Addition of Transformer model training script.
- **23-05**: Released the new Global Normalised CAM utility notebooks.
- **23-03**: Update on dataset access for external collaborators.

## Contributing

Contributions to this project are welcome! Please refer to the following guidelines:
- For bug reports or feature requests, please open an issue.
- For direct contributions, please fork the repository, make your changes, and submit a pull request.

## License and Citation

This work is licensed under a Creative Commons Attribution 4.0 International License. If you use this tool or dataset in your research, please cite it as follows:

```bibtex
@article{YourWork2024,
  title={Sketch-supervised Histopathology Tumour Segmentation: Dual CNN-Transformer with Global Normalised CAM},
  author={Your Name and Collaborator Names},
  journal={Journal or Conference Name},
  year={2024}
}
```

## Acknowledgements

This project is supported by [Your University/Institution Name]. We thank our colleagues who contributed to this project and provided insights and expertise.
