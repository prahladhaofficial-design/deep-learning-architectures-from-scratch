# Sequence Learners


Project 4: Building Sequence Learners from Scratch
Theoretical Analysis & Ablation Studies
prahladha
4.1 & 4.3 Forward Pass Formulations
Vanilla RNN
For a sequence of inputs x , the recurrent state and output for a many-to-one configuration are
t
defined as:
h = tanh(W x + W h + b )
t xh t hh t1 h
y = softmax(W h + b )
T hy T y
where h = 0.
0
Long Short-Term Memory (LSTM)
To combat vanishing gradients, the LSTM uses a cell state C and gating mechanisms:
t
f = (W [h , x ] + b ) (Forget Gate)
t f t1 t f
i = (W [h , x ] + b ) (Input Gate)
t i t1 t i
C  = tanh(W [h , x ] + b ) (Candidate State)
t C t1 t C
C = f  C + i  C  (Cell State Update)
t t t1 t t
o = (W [h , x ] + b ) (Output Gate)
t o t1 t o
h = o  tanh(C ) (Hidden State)
t t t
4.2 Part 2: Backpropagation Through Time (BPTT)
In an RNN, the weight matrices (e.g., W = W ) are shared across all time steps. Therefore, the
hh
total gradient of the loss L with respect to W is the sum of the gradients at each time step. Using
the multivariate chain rule:

T T
L (cid:88) L (cid:89) h j h k
=
W h h W
T j1
k=1 j=k+1
The critical component here is the Jacobian matrix
hj
. For a Vanilla RNN:
hj1
h
j = W T diag(1  tanh2(z ))
h hh j
j1
1


where z = W x + W h + b . Over long sequences, computing the product
(cid:81)T hj
j xh j hh j1 h j=k+1 hj1
requires multiplying the weight matrix W many times. If the maximum eigenvalue of W is
hh hh
greater than 1, the gradients will explode (grow exponentially). To prevent this, we apply Gradient
Clipping, which bounds the gradient norm to a maximum threshold.
4.4 Part 4: Ablation Studies (Theoretical Answers)
Ablation A: The Vanishing Gradient in Action
Technical focus: Explain your observations based on the RNN gradient flow equation. Why does the
gradient vanish in the Vanilla RNN but not the LSTM?
Analysis: In the Vanilla RNN, the gradient flow through time depends on the continuous
multiplication of the term W T diag(tanh(z )). Since the derivative of the tanh function is bounded
hh j
(cid:13) (cid:13)
between (0, 1], and weights are usually initialized to be small, the norm of the Jacobian (cid:13) hj (cid:13)
(cid:13) hj1 (cid:13)
is typically less than 1. When this is multiplied T  k times over a long sequence, the gradient
exponentially decays to zero (vanishing gradient). Thus, the network fails to update weights based
on long-term dependencies (e.g., at t = 1).
In the LSTM, the gradient flows primarily through the cell state C . The update equation is
t
C = f  C + i  C  . Taking the derivative:
t t t1 t t
C
t
= f + . . .
t
C
t1
This additive structure forms a Constant Error Carousel (CEC). If the network learns to set
the forget gate f  1 (meaning remember the past), the local derivative Ct becomes ap-
t Ct1
proximately 1. Consequently, the gradient can flow backwards through many time steps without
exponentially decaying, allowing LSTMs to capture long-term dependencies.
Ablation B: Coupled Input and Forget Gates
Modification: C = f  C + (1  f )  C
t t t1 t t
Technical focus: Compare the training time and final accuracy. Does reducing the parameter count
improve or hurt performance on your dataset?
Expected Theoretical Outcome: By coupling the gates, we force the network to only write
new information (C  ) exactly at the locations where it erases old information (C ).
t t1
 Training Time: Training time per epoch will decrease because we completely remove the
computation and parameters of the input gate i (W , b ).
t i i
 Performance Impact: On relatively simple or moderately complex tasks like IMDb Sen-
timent Analysis, reducing the parameter count often acts as a form of regularization. It
prevents overfitting and generally performs equally well or slightly better than the stan-
dard LSTM. However, on highly complex generative tasks where the model needs independent
control over what to erase and what to write, restricting the hypothesis space by coupling
gates could slightly hurt the final capacity and performance.
2


Ablation C: Optimization Strategies
Technical focus: How does Adam compare to standard SGD when optimizing the complex loss surface
of an LSTM?
Analysis: Recurrent neural networks typically have highly non-convex loss surfaces filled with
sharp ravines, plateaus, and saddle points due to the repeated application of the same weight
matrices.
 Standard SGD updates parameters uniformly scaled by a single learning rate. In ravines,
SGD tends to oscillate wildly across the steep slopes while making agonizingly slow progress
along the flat optimum path. It is also highly sensitive to the vanishing gradient, often getting
stuck on plateaus.
 Adam (Adaptive Moment Estimation) computes individual adaptive learning rates for dif-
ferent parameters. By keeping an exponentially decaying average of past gradients (first
moment, providing momentum) and squared gradients (second moment, normalizing vari-
ance), Adam effectively dampens oscillations in steep directions and accelerates movement
along flat directions.
Therefore, Adam converges significantly faster and more reliably on the complex loss surface of an
LSTM compared to standard SGD.
3
