# **PixelRNN with Diagonal BiLSTM - README**

## **Project Overview**
This repository contains an implementation of **PixelRNN with Diagonal BiLSTM** for image generation. The model is trained on the **CIFAR-10** and **MNIST** datasets using **autoregressive modeling**, where each pixel is predicted sequentially based on previously generated pixels.

## **Features**
- **PixelRNN with Diagonal BiLSTM** for capturing spatial dependencies.
- **Masked convolutions** ensuring causal dependencies in pixel prediction.
- **Dropout and weight decay** for regularization and improved generalization.
- **Grid search for hyperparameter tuning** (learning rate, dropout, weight decay, etc.).
- **Training monitoring using TensorBoard**.

## **Directory Structure**
```
├── models                  # Saved model checkpoints
├── runs/                   # TensorBoard logs
├── Binary\ PixelRNN.py     # Principal code
├── hp_search.py     # Hyparameter searching code
├── hyperparameter_results.csv  # Grid search results
└── README.md               # Project documentation
```

## **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/PixelRNN_Diagonal_BiLSTM.git
cd PixelRNN_Diagonal_BiLSTM
```

### **2️⃣ Create Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # On Linux/macOS
env\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

## **Training & Evaluation**
### **1️⃣ Train the Model**
```bash
python train.py --dataset cifar10 --epochs 100 --batch_size 16 --lr 0.005
```
Options:
- `--dataset` : `cifar10` or `mnist`
- `--epochs` : Number of epochs
- `--batch_size` : Batch size
- `--lr` : Learning rate

### **2️⃣ Monitor Training with TensorBoard**
```bash
tensorboard --logdir=runs --port=6006
```
Then open **http://localhost:6006/** in your browser.


## **Hyperparameter Tuning**
A grid search was performed to identify the best hyperparameters:
| Learning Rate | Dropout | Weight Decay | Validation Loss |
|--------------|---------|--------------|----------------|
| 0.005       | 0.3     | 1e-6         | **0.1596**     |
| 0.005       | 0.5     | 1e-6         | 0.1610         |

The best configuration found is **learning rate = 0.005, dropout = 0.3, weight decay = 1e-6**.

## **Key Findings**
- The **Diagonal BiLSTM captures long-range dependencies** effectively.
- **Dropout (0.3 - 0.5) and weight decay (1e-6) help prevent overfitting**.
- The best test loss achieved was **7 Bits/Dim**, though still behind the PiwelRNN paper results (~3 Bits/Dim) by Google DeepMind.

## **Future Improvements**
- Extend the model to **PixelCNN** for faster generation.
- Implement **larger architectures (deeper LSTMs)** for improved results.
- Try **alternative datasets** beyond CIFAR-10 and MNIST.


