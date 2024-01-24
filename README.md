# CNN for Image Recognition

The code shows a detailed workflow for working with convolutional neural networks (CNNs) using the CIFAR-10 dataset(https://www.cs.toronto.edu/~kriz/cifar.html). It initiates by loading the dataset and transforming class labels into one-hot encoded vectors. Subsequently, the training set undergoes shuffling and partitioning into training and validation subsets. An initial CNN model is constructed and compiled with Adagrad as the chosen optimizer. This model is then trained on the training set, and the training history is visually represented through a loss curve.

Following this, the model undergoes recompilation and is trained on the complete training set. The model's performance is evaluated on the testing set, providing insights into its generalization capabilities. Additionally, a new CNN model is introduced, featuring added dropout layers for regularization purposes. The architecture of this new model is displayed using the summary method.

In conclusion, the code provides a comprehensive and sequential example of the entire process involved in working with CNNs for image classification tasks, covering data preprocessing, model construction, compilation, training, evaluation, and architectural exploration.

Reference: https://arxiv.org/pdf/1502.03167.pdf

