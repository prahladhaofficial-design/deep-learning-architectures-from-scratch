# Deep Learning Project2


Project 2
Deadline: 20th April, 2026
Project Version: (<roll-number>.ipynb) Total: 200 Evaluation Points
Please read the following instructions before starting the project:
 It is strongly recommended that you work on this homework on an individual basis. If you have
any modules or concerns, feel free to contact the instructor or the reviewers at least 2 days before deadline.
 You can also use markdown cells in the notebook to explain theoretical problems and observations.
 Ensure that all submitted code runs without errors. Non-executable code will receive zero evaluation points for
the corresponding module.
1 Numpy Implementation of CNN (15 Evaluation Points)
In this project, you will implement each of the components of a Convolutional Neural Network (CNN)
from scratch (i.e., without using built-in functions for convolution, pooling, non-linearity, padding, or strid-
ing). Your implementation must accept an image input and generate an output vector. Use random weights
for filter kernels and fully connected layers.
i) Convolution function: It accepts as input an image, a filter kernel, stride, padding, and the non-
linear function. The function must correlate (convolve) the input image (after padding if specified) with
the kernel (at the specified stride size) and generate an output activation after applying the specified
non-linearity. Verify with standard non-linear activation functions such as sigmoid, tanh, ReLU, and
Parametric ReLU (PReLU). Display the input image, the filter kernel, and the output activation map.
Ensure that your function can accept multi-channel input and a corresponding kernel volume. (3)
ii) Pooling function: It accepts as input the activation map output from the convolution function,
a pooling function, and stride. The function must output the appropriately pooled activation map.
Display the input activation map and the pooled output. (2)
iii) Convolution layer function: It accepts as input a volume (image or activation maps), filter kernels,
stride, padding, and the non-linear function. The function must convolve the input volume (after
padding if specified) with each of the kernels (at the specified stride size) and generate an output
activation volume after applying the specified non-linearity. Display the input image or activation maps,
filter kernels, and output activation maps. Verify that the output has the expected size (W  H  C).
(1)
iv) Pooling layer function: It accepts as input the activation map volume, the pooling function, and
stride, and generates a pooled output volume. Display the input and output volumes. (1)
v) Flattening function: It accepts as input the activation map volume output by the pooling layer and
generates a vector of a specified size. Note that this function has a weight matrix associated with it
whose size must match the input and desired output sizes. (1)
1


vi) Multilayer Perceptron (MLP): It accepts as input a vector, the number of hidden layers, the size
of each hidden layer, the non-linear function, and the size of the output layer. Generate the output
vector both with and without applying the softmax function at the output layer. (2)
vii) Feed-forward CNN architecture: Use the above functions to implement a CNN with the following
architecture:
 Input image of size 32  32  3 (CIFAR-10 dataset)
 Convolution layer with 4 kernels of size 5  5, ReLU activation, stride 1, and same spatial dimen-
sions as input
 Max pooling layer of size 2  2 with stride 2
 Convolution layer with 4 kernels of size 5  5  4, ReLU activation, stride 1, and same spatial
dimensions
 Max pooling layer of size 2  2 with stride 2
 Flattening layer (identity mapping)
 MLP with one hidden layer of 49 nodes and an output layer of 10 nodes
 ReLU activation and softmax output
Verify that the network accepts an image input and outputs a vector of appropriate dimension. (5)
2 PyTorch Implementation of CNN (25 Evaluation Points)
a) Train the CNN described in the NumPy CNN architecture for image classification using PyTorch. Initialize all
network parameters randomly. (2)
b) Use cross-entropy loss to compute the error at the softmax output layer. Note that the ground-truth
labels are one-hot encoded vectors of dimension 10. (9)
i) Vanilla SGD with learning rate  = 0.001.
ii) Momentum with  = 0.001,  = 0.9.
iii) RMSProp with  = 0.001,  = 0.9.
For training:
 Choose 100 images per class from the training set (total 1000 images).
 Experiment with different mini-batch sizes.
 Shuffle the training data after each epoch.
 Train for 15 epochs (you may experiment with more epochs).
c) Perform the following after each epoch: (5)
i) Compute the error on training and test datasets. Use 10 images per class from the test set (total
100 images). Plot training and test error as a function of epochs.
ii) Visualize activation maps by selecting representative slices from the activation volumes of each
convolution layer.
iii) Report the classification accuracy.
d) Use tSNE to visualize the bottleneck feature at the end of the first epoch and the last epoch. Use the
same set of test images as in the previous module. (3)
2


e) Apply dropout at the hidden layer of the MLP with dropout rates p = 0.2, 0.5, 0.8. Train using
RMSProp and report results. Compare performance with the model without dropout and comment
on differences in training and testing performance. (3)
f) Introduce batch normalization (BN) at the hidden layer of the MLP and report the performance. Batch
normalization should be applied at the output of the hidden layer before applying the non-linearity.
In your implementation, include both BN and dropout (with p = 0.2). Compare the performance with
the vanilla implementation as well as the model with only dropout, and comment on the training and
testing performance. (3)
3 Backpropagation in CNN (Theory) (20 evaluation points)
Consider the convolutional neural network described below. The input is an 8  8 grayscale image X.
 A 3  3 convolutional filter with bias, stride = 1, and padding = 1 is applied to produce feature map
Z .
1
 A ReLU activation is applied to obtain A .
1
 A 2  2 max-pooling layer with stride = 2 is applied to obtain P .
1
 A second 33 convolutional layer with 2 filters, stride = 1, and no padding is applied on P , producing
1
Z .
2
 A ReLU activation is applied to obtain A .
2
 The feature maps are flattened into a vector F , which is passed through a fully connected layer (with
weights W , bias b) to produce Z .
3
 A Sigmoid activation is applied to obtain the final output y.
The network is trained using binary cross-entropy (BCE) loss.
Analysis focus:
(i) Draw a clear flow diagram of the network described above. (2)
(ii) Determine the dimensions of the following: (3)
Z , A , P , Z , A , F, Z , y
1 1 1 2 2 3
(iii) Compute the total number of trainable parameters in the network . (3)
(iv) Using backpropagation, derive the gradients of the loss with respect to: (2+4+6)
 Fully connected layer parameters (W, b)
 Second convolutional layer parameters
 First convolutional layer parameters
4 Building Sequence Learners from Scratch: Vanilla RNNs, BPTT, and LSTMs (70 evaluation points)
In this Module, you will build sequence learning models entirely from scratch using only standard Python
and NumPy. You will implement a Vanilla Recurrent Neural Network (RNN) and a Long Short-Term Memory
(LSTM) network to solve a many-to-one sequence task, such as Sentiment Analysis. You will code the
forward passes, derive and implement Backpropagation Through Time (BPTT), and perform ablation studies
to understand the internal mechanisms, vanishing gradients, and the effects of architectural variations.
3


Part 0: Dataset, Input Representation, and Evaluation Protocol
To ensure consistency across implementations, all students must follow the specifications below.
Dataset: You will use the IMDb Movie Reviews Dataset for binary sentiment classification. You
are not required to use the full dataset. You can do the experiments using a subset of the dataset as well.
 Each sample consists of a movie review (a sequence of words) and a label:
 y = 1 (positive sentiment)
 y = 0 (negative sentiment)
 You may use a preprocessed version of the dataset (tokenized and cleaned), but you must clearly
document all preprocessing steps.
 Split the dataset into training, validation, and test sets (e.g., 80/10/10).
Input Representation: Since RNNs operate on numerical vectors, you must convert text into vector
representations. You have to use both of the following approaches and report any differences that you may
find:
1. One-Hot Encoding:
 Represent each word x as a one-hot vector of size |V |, where V is the vocabulary.
t
 This leads to high-dimensional sparse inputs.
2. Learned Word Embeddings (Recommended):
 Initialize an embedding matrix E  R|V |d.
 Each word is mapped to a dense vector:
x = E[w ]
t t
where w is the index of the word at time step t.
t
 The embedding matrix should be trained jointly with the model.
Sequence Handling:
 Fix a maximum sequence length T (e.g., T = 50 or T = 100).
 Truncate longer sequences and pad shorter ones with a special token.
Evaluation Metrics: To evaluate model performance, report the following metrics on the test set:
1. Accuracy
2. F1-Score
Reporting Requirements:
 Report Accuracy and F1-score for both:
 Vanilla RNN
 LSTM
 Compare results across:
 Different sequence lengths
 Optimizers (SGD vs Adam)
 Include loss curves and discuss convergence behavior.
4


4.1 Part 1: Vanilla RNN and the Forward Pass (12)
Your first task is to implement the standard Recurrent Neural Network architecture. RNNs are designed
to handle variable-length inputs by applying the same transformation at every time step, modeling the
dependence among the inputs.
Tasks:
1. Initialize the hidden state at t = 0 as h = 0. (2)
0
2. Implement the recurrent state update for a sequence of inputs x : (7)
t
h = tanh(W  x + W  h + b )
t xh t hh t1 h
3. Implement the output layer for a many-to-one configuration (e.g., predicting the final sentiment at the
end of the sequence): (3)
y = softmax(W  h + b )
t hy t y
4.2 Part 2: Backpropagation Through Time (BPTT) and Gradient Issues (18)
To train the network, you must implement BPTT. The error is a scalar function of the sequence, and the
computation shares the same parameters across time, meaning you must pool the gradients.
Tasks:
1. Implement the BPTT algorithm. You will need to compute gradients L , L , and L . (6)
V W U
2. Code the gradient calculation for the shared weight W accumulating over time steps. Use the chain
rule formulation provided in the lectures for a sequence of length T : (6)

T T
L = (cid:88) L  (cid:89) h j  h k
W h h W
T j1
k=1 j=k+1
3. Exploding Gradients: Because the product (cid:81)T hj can lead to exploding gradients, imple-
j=k+1 hj1
ment Gradient Clipping to prevent your loss from returning NaN values. (6)
4.3 Part 3: Long Short-Term Memory (LSTM) Implementation (20)
Vanilla RNNs struggle to learn long-term dependencies because the gradient vanishes when the product of
the derivatives hj shrinks over long sequences. To fix this, you will implement an LSTM, which utilizes
hj1
a cell state C to store long-term information and gates to erase, write, and read information.
t
Tasks: Implement the LSTM node and its gates, ensuring all relevant outputs are bounded between [0, 1]
using a sigmoid () function where appropriate. Use the concatenated vector [h , x ] for your computations.
t1 t
1. Forget Gate: Decide what to throw away from the cell state. (2)
f = (W  [h , x ] + b )
t f t1 t f
2. Input Gate & Candidate State: Decide what new information to store. (4)
i = (W  [h , x ] + b )
t i t1 t i
C  = tanh(W  [h , x ] + b )
t C t1 t C
5


3. Cell State Update: Combine the forget and input gates to update the memory. (6)
C = f  C + i  C
t t t1 t t
4. Output Gate & Hidden State: Compute the output and the next hidden state. (8)
o = (W  [h , x ] + b )
t o t1 t o
h = o  tanh(C )
t t t
(Hint: Verify during your backward pass that the gradient L does not vanish because ht depends on a
diagonal matrix D(o  (C )), allowing gradients to flow uni  m ht peded through the cell state C C t ).
t t t
4.4 Part 4: Ablation Studies (20)
Once you have built both models from scratch, run the following ablations to demonstrate your understanding
of the architectures and optimizers.
Ablation A: The Vanishing Gradient in Action (5)
 Experiment with sequence lengths of T = 5, 20, and 50.
 Plot the magnitude of the gradient L for both the Vanilla RNN and the LSTM.
h1
 Technical focus: Explain your observations based on the RNN gradient flow equation. Why does the
gradient vanish in the Vanilla RNN but not the LSTM?
Ablation B: Coupled Input and Forget Gates (10)
 Modify your LSTM implementation to use Coupled Input and Forget Gates.
 Instead of computing an independent input gate i , force the network to only input new information
t
when it forgets old information.
 Replace your standard Cell State update with:
C = f  C + (1  f )  C
t t t1 t t
 Technical focus: Compare the training time (epochs to convergence) and final accuracy of the standard
LSTM vs. the Coupled Gate variant. Does reducing the parameter count improve or hurt performance
on your dataset?
Ablation C: Optimization Strategies (5)
 Train your networks using standard Stochastic Gradient Descent (SGD), where w = w L(w, x , y ).
w i i
 Train your networks using the Adam optimizer, ensuring you implement the bias correction step since
the estimates start from 0.
 Technical focus: Plot the loss curves. How does Adam compare to standard SGD when optimizing the
complex loss surface of an LSTM?
5 Building a Transformer Encoder from Scratch: Attention, Training, and Analysis (40
evaluation points)
In this module, you will implement the core components of a Transformer encoder using only Python and
NumPy. You will build a self-attention mechanism, construct a single-layer Transformer encoder, and apply
it to a sequence classification task (e.g., sentiment classification). You will also analyze attention behavior
and optimization.
6


Part 0: Dataset and Setup
You will use the IMDb Movie Reviews Dataset for binary sentiment classification. You are not required
to use the full dataset. You can do the experiments using a subset of the dataset as well.
 Tokenize text and map words to indices.
Input Representation:
 Use learned embeddings E  R|V |d.
 Add positional encodings to retain order information.
5.1 Part 1: Scaled Dot-Product Attention (10)
Implement the core attention mechanism.
Tasks:
1. Compute queries, keys, and values:
Q = XW , K = XW , V = XW
Q K V
2. Implement scaled dot-product attention:
(cid:18) QKT (cid:19)
Attention(Q, K, V ) = softmax  V
d
k
3. Verify that attention weights sum to 1 across each row.
5.2 Part 2: Multi-Head Attention (5)
1. Split Q, K, V into h heads.
2. Compute attention independently for each head.
3. Concatenate outputs and apply a final linear projection.
5.3 Part 3: Transformer Encoder Block (10)
Build a single Transformer encoder layer.
Components:
 Multi-head attention
 Feedforward network:
FFN(x) = max(0, xW + b )W + b
1 1 2 2
 Residual connections
 Layer normalization
Task:
 Implement a full forward pass of the encoder block.
7


5.4 Part 4: Sequence Classification (4)
 Use the final representation of a special [CLS] token or mean-pooling.
 Pass it through a linear layer + softmax for classification.
5.5 Part 5: Training and Optimization (6)
 Train using cross-entropy loss.
 Implement:
 SGD
 Adam (with bias correction)
 Plot loss curves.
5.6 Part 6: Analysis and Ablations (5)
Ablation A: Effect of Number of Heads
 Compare h = 1 vs h = 4
 Report accuracy and training behavior
Ablation B: Positional Encoding
 Remove positional encoding
 Explain performance change
6 Long-Context Modeling: From RNNs to Selective State Space Models (30 evaluation points)
We consider a modified version of the IMDb dataset called IMDb-Long. To ensure consistency and eliminate
label leakage, you must construct the IMDb-Long dataset as follows:
Neutral Text Source (Strictly Neutral)
Use the WikiText-2 dataset:
 datasets.load dataset("wikitext", "wikitext-2-raw-v1")
Extract sentences from WikiText-2 as neutral filler.
Rationale:
 Wikipedia text is largely factual and topic-diverse
 It is independent of IMDb sentiment labels
 It contains minimal sentiment bias compared to reviews
8


IMDb-Long Construction
For each IMDb review:
1. Tokenize the review and split it into three parts:
review = (P , P , P )
1 2 3
2. Discard P .
2
3. Sample random sentences from WikiText-2 and use them as filler.
4. Construct the final sequence:
final = P || neutral filler || P
1 3
5. Append additional neutral sentences if needed to reach target length:
 Experiment with each of 256, 512, or 1024 tokens.
6. Keep the original IMDb label unchanged.
Important Constraints
 Neutral filler must come only from WikiText-2
 Do not reuse IMDb sentences as filler
 Ensure tokenizer is identical across all models
 You may use whatever subset of IMDb you used for Q5
 Use fixed random seed for reproducibility
6.1 Part 1: Computational Scaling (5)
Let the sequence length be L and hidden dimension be d.
 Derive the time complexity per layer of a Transformer encoder using self-attention. (2)
 Derive the per-sequence time complexity of a standard RNN. (2)
 For L = 1024, d = 256, compute the ratio: Transformer cost (1)
RNN cost
6.2 Part 2: Gradient Propagation in RNNs (5)
Consider the recurrence:
h = Ah + Bx
t t1 t
 Express h in terms of h and {x , . . . , x }. (2)
t 0 1 t
 Show that:
h
t = Ak
h
tk
(2)
 Let (A) be the spectral radius: (1)
 What happens if (A) > 1?
 What happens if (A) < 1?
9


6.3 Part 3: State Space Models (SSM) (6)
Consider the discrete SSM:
h = Ah + Bx , y = Ch
k k1 k k k
 Show that:
k
(cid:88)
y = CAkiBx
k i
i=0
(3)
 Interpret this as a 1D convolution with kernel:
K = CAkiB
ki
(2)
 Explain why this system is time-invariant and state one limitation. (1)
6.4 Part 4: Selective State Space Models (6)
We now introduce a simple selective mechanism.
h = Ah + Bx
k k1 k
y = g  (Ch ), g = (W x )
k k k k g k
 Verify that all dimensions are consistent. (2)
 Explain why this model is no longer time-invariant. (2)
 Show that:
(cid:32) k (cid:33)
(cid:88)
y = g  CAkiBx
k k i
i=0
Explain why this cannot be represented as a fixed convolution. (2)
6.5 Part 5: Implementation and Experiments (8)
You must implement and compare the following models:
 LSTM
 Transformer Encoder
 Linear SSM
 Selective SSM
10


6.5.1 (a) Linear SSM Implementation
Complete the following:
1 class LinearSSM ( nn . Module ):
2 def __init__ ( self , d_in , d_hidden , num_classes ):
3 super () . __init__ ()
4 self .A = nn . Parameter ( torch . randn ( d_hidden , d_hidden ) * 0.1)
5 self .B = nn . Linear ( d_in , d_hidden )
6 self .C = nn . Linear ( d_hidden , d_hidden )
7 self . fc = nn . Linear ( d_hidden , num_classes )
8
9 def forward ( self , x):
10 batch , seq_len , _ = x. shape
11
12 # Step: initialize hidden state
13 h = None
14
15 for t in range ( seq_len ):
16 x_t = x [: , t]
17
18 # Step: recurrence
19 # h = A h + B x_t
20 pass
21
22 # Step: output
23 y = None
24 out = None
25 return out
6.5.2 (b) Selective SSM Implementation
1 class SelectiveSSM ( nn . Module ):
2 def __init__ ( self , d_in , d_hidden , num_classes ):
3 super () . __init__ ()
4 self .A = nn . Parameter ( torch . randn ( d_hidden , d_hidden ) * 0.1)
5 self .B = nn . Linear ( d_in , d_hidden )
6 self .C = nn . Linear ( d_hidden , d_hidden )
7
8 # Step: define gating network
9 self . W_g = None
10
11 self . fc = nn . Linear ( d_hidden , num_classes )
12
13 def forward ( self , x):
14 batch , seq_len , _ = x. shape
15
16 # Step: initialize hidden state
17 h = None
18
19 for t in range ( seq_len ):
20 x_t = x [: , t]
21
22 # Step: update state
23 # h = ...
24
11


25 # Step: compute gate
26 g = None
27
28 # Step: selective output
29 y = None
30
31 # Step: classification
32 out = None
33 return out
6.5.3 (c) Experiments
Train all models for sequence lengths {256, 512, 1024}.
Report:
 Accuracy
 Training time per epoch
Plot:
 Accuracy vs sequence length
(d) Analysis
Analysis focus:
 Which model degrades most with longer sequences?
 Does Selective SSM outperform LSTM or Transformer?
 Give one failure case of Transformer on long sequences.
12
