# deep_learning_course
Repository for the learning materials of the course 'Deep Learning Specialization' by DeepLearning.ai on coursera.

# Course Structure

The course 'Deep Learning Specialization' consist of 5 courses:

- Neural Networks and Deep Learning
- Improving Deep Neural Networks
- ML Strategy
- Convolutional Neural Networks
- Sequence Models

# Key Learnings

*Neural Networks and Deep Learning*

- Deep and Shallow neural network architectures
- Forward propagation computing linear and nonlinear transformations in each hidden layer
- Types of AFs (Sigmoid, Tanh, [leaky] ReLU, SoftMax) and need for non-linearity
- Derivatives of standard AFs, need of non-zero initialization
- Backward propagation calculating derivatives for each layer using chain rule of derivation
- Gradient Descent algorithm
- Vectorized implementation of L-layers deep neural networks (forward and backward propagation)
- Need of depth in neural networks and progressive ‘feature hierarchy’ learning in hidden layers

*Improving Deep Neural Networks*

- Setting up sizes of Train/Dev/Test sets for small and large datasets and interpretation of model performance on them
- Interpretation of Bias and Variance in model and determining direction to diagnose performance
- Guidance on choosing hyperparameters and their scale appropriately for hyperparameter tuning 
- Regularization techniques (L1/L2 regularization, dropout, early stopping, augmentation) to address
overfitting (memorization)
- Normalization of input and effect on cost function
- Weight initialization techniques (He, Glorot initializers) to address ‘Vanishing/Exploding gradients problem’ in deep
neural networks
- Optimization algorithms (Variations in Gradient Descent)
  - Stochastic/Minibatch/Batch Gradient Descent, effect on computation and convergence
  - Gradient Descent with momentum for faster convergence preventing oscillations
  - RMSProp adaptively adjusts learning rates for each parameter by amplifying smaller gradients and damping larger 
  ones, ensuring balanced, and stable optimization
  - Adam combines momentum and RMSProp, using adaptive learning rates and bias correction for efficient, stable, and 
  fast optimization
  - Learning Rate decay adjusting learning rate to improve convergence near optima

- Weighted average and bias correction concepts and their applications in time series data (i.e. sensor data, weather
forecast, finance)
- Problem of local optima (convex or concave), saddle points (convex and concave) and plateaus (neither concave 
nor convex)
- Batch Normalization to counter Covariate shift problem and stabilize learning
- SoftMax classifier for multilabel learning
- Concept of computation graph and automatic differentiation to calculate gradients of custom loss/cost functions using
Gradient Tape

*ML Strategy*

- Setting up a clear model goal by defining optimizing and satisfying metrics
- Orthogonalization framework to tune hyperparameters systematically
- Bias/Variance analysis
- Human level performance and Bayes optimal error
- Error analysis to differentiate sources of errors and approaches to address them
- Defining Train/Dev/Test set distributions and size
- Training and testing on different distributions
- Addressing mismatched distributions in train and dev/test set and techniques for transfer learning
- Multitask Learning, Transfer Learning, End-to-End Learning

*Convolutional Neural Networks* (Contd.)

- ...

