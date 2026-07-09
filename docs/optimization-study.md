# Optimization Study


Backpropagation in CNN
prahladha
Module
Consider the convolutional neural network described below. The input is an 8  8 grayscale
image X.
 A 3  3 convolutional filter with bias, stride = 1, and padding = 1 is applied to produce
feature map Z .
1
 A ReLU activation is applied to obtain A .
1
 A 2  2 max-pooling layer with stride = 2 is applied to obtain P .
1
 A second 3  3 convolutional layer with 2 filters, stride = 1, and no padding is applied on
P , producing Z .
1 2
 A ReLU activation is applied to obtain A .
2
 The feature maps are flattened into a vector F , which is passed through a fully connected
layer (with weights W , bias b) to produce Z .
3
 A Sigmoid activation is applied to obtain the final output y.
The network is trained using binary cross-entropy (BCE) loss.
Analysis focus:
(i) Draw a clear flow diagram of the network described above.
(ii) Determine the dimensions of the following: Z , A , P , Z , A , F, Z , y
1 1 1 2 2 3
(iii) Compute the total number of trainable parameters in the network.
(iv) Using backpropagation, derive the gradients of the loss with respect to:
 Fully connected layer parameters (W, b)
 Second convolutional layer parameters
 First convolutional layer parameters
1


Solutions
(i) Flow Diagram of the Network
Input Image X
(8  8  1) ReLU Activation Feature Map A 2
Conv Layer 1 Feature Map Z 2 Flatten Operation
Filter: 3  3, S = 1, P = 1
Conv Layer 2
Flattened Vector F
Feature Map Z Filters: 2  (3  3), S = 1, P = 0 BCE Loss L
1
Fully Connected Layer
ReLU Activation Feature Map P 1 Weights W , Bias b Prediction y
Max Pooling Linear Output Z Sigmoid Activation
Feature Map A 3
1 Size: 2  2, S = 2
(ii) Dimensions of Feature Maps and Vectors
The spatial dimension of an output feature map for a convolutional or pooling layer is given by
the general formula:
(cid:22) (cid:23)
I  K + 2P
O = + 1
S
where I is the input dimension, K is the filter (kernel) size, P is the padding, and S is the
stride. The number of output channels is equal to the number of filters applied.
 Dimension of Z : The input X has spatial dimensions I = 8 and 1 channel. For the
1
first convolutional layer, the filter size K = 3, padding P = 1, stride S = 1, and there is
1 filter.
(cid:22) (cid:23) (cid:22) (cid:23)
8  3 + 2(1) 7
O = + 1 = + 1 = 8
Z1
1 1
Since there is 1 filter, the dimension is 8  8  1.
 Dimension of A : The ReLU activation function (f (x) = max(0, x)) is applied element-
1
wise to Z . It does not alter the spatial dimensions or the number of channels. Therefore,
1
the dimension is 8  8  1.
 Dimension of P : For the max-pooling layer, the input is A (I = 8). The pooling
1 1
window size K = 2, stride S = 2, and padding P = 0.
(cid:22) (cid:23) (cid:22) (cid:23)
8  2 + 2(0) 6
O = + 1 = + 1 = 3 + 1 = 4
P1
2 2
Pooling operates on each channel independently. Thus, the dimension is 4  4  1.
2


 Dimension of Z : The input to this layer is P (I = 4). The filter size K = 3, padding
2 1
P = 0, stride S = 1, and there are 2 filters.
(cid:22) (cid:23) (cid:22) (cid:23)
4  3 + 2(0) 1
O = + 1 = + 1 = 2
Z2
1 1
Because there are 2 distinct filters, the output will have 2 channels. The dimension is
2  2  2.
 Dimension of A : Applying element-wise ReLU to Z leaves the dimensions unchanged.
2 2
Therefore, the dimension is 2  2  2.
 Dimension of F : The flattening step reshapes the 3D tensor A into a 1D column vector.
2
The total number of elements is 2  2  2 = 8. Therefore, the dimension is 8  1.
 Dimension of Z : The fully connected layer maps the 8  1 vector F to a single scalar
3
(since we are doing binary classification indicated by the BCE loss and single Sigmoid
output). Therefore, the dimension is 1  1.
 Dimension of y: The Sigmoid activation is applied to the scalar Z , producing a single
3
probability value. Therefore, the dimension is 1  1.
Final Result for Dimensions:
Z : 8  8  1 A : 8  8  1 P : 4  4  1 Z : 2  2  2
1 1 1 2
A : 2  2  2 F : 8  1 Z : 1  1 y : 1  1
2 3
(iii) Total Number of Trainable Parameters
The number of trainable parameters in a convolutional layer is given by:
Parameters = (Filter Height  Filter Width  Input Channels + 1)  Number of Filters
The +1 accounts for the bias term for each filter.
1. First Convolutional Layer (Conv1):
Filter Size = 3  3
Input Channels (from X) = 1
Number of Filters = 1
Weights per filter = 3  3  1 = 9
Bias = 1
Total for Conv1 = (9 + 1)  1 = 10
2. Second Convolutional Layer (Conv2):
Filter Size = 3  3
Input Channels (from P ) = 1
1
Number of Filters = 2
Weights per filter = 3  3  1 = 9
Bias per filter = 1
Total for Conv2 = (9 + 1)  2 = 20
3


3. Fully Connected Layer (FC): The fully connected layer formula is (Input Size
Output Size) + Output Size (for bias).
Input Size (from F ) = 8
Output Size (to Z ) = 1
3
Weights W = 8  1 = 8
Bias b = 1
Total for FC = 8 + 1 = 9
To find the overall total, we sum the parameters from all layers:
Total Trainable Parameters = 10 (Conv1) + 20 (Conv2) + 9 (FC) = 39
Final Result for Trainable Parameters:
Total Parameters = 39
(iv) Backpropagation and Gradients Derivation
Let the ground truth binary label be y  {0, 1}. The Binary Cross-Entropy (BCE) Loss L is
defined as:
L =  [y log y + (1  y) log(1  y)]
The predicted probability y is obtained via the Sigmoid activation:
1
y = (Z ) =
3 1 + eZ3
Step 0: Gradient of Loss with respect to Output Z
3
We apply the chain rule: L = L y . First, we find the derivative of the loss with respect to
Z3 y Z3
y:
(cid:18) (cid:19)
L 1 1
=  y  + (1  y)
y y 1  y
y 1  y
=  +
y 1  y
y(1  y) + y(1  y)
=
y(1  y)
y + yy + y  yy
=
y(1  y)
y  y
=
y(1  y)
Next, the derivative of the Sigmoid function:
y
= (Z )(1  (Z )) = y(1  y)
3 3
Z
3
Multiplying them together gives the local gradient  :
Z3
(cid:18) (cid:19)
L y  y
 = =  y(1  y) = y  y
Z3
Z y(1  y)
3
4


1. Gradients for Fully Connected Layer Parameters (W, b)
The forward pass for the fully connected layer is defined as:
8
(cid:88)
Z = W F + b
3 i i
i=1
where W is the weight vector [W , W , . . . , W ] and F is the input vector [F , F , . . . , F ]T .
1 2 8 1 2 8
Using the chain rule, the gradient with respect to each weight W is:
i
Z
3
= F
i
W
i
L L Z
3
= =  F = (y  y)F
W Z W
Z3 i i
i 3 i
The gradient with respect to the bias b is:
Z
3
= 1
b
L L Z
3
= =   1 = y  y
b Z b
Z3
3
To propagate the error backwards to the previous layers, we calculate the gradient with
respect to the input features F :
i
Z
3
= W
i
F
i
L L Z
3
 = = =  W = (y  y)W
Fi
F Z F
Z3 i i
i 3 i
Final Result for Fully Connected Layer Parameters:
L L
= (y  y)F T where scalar components are = (y  y)F
i
W W
i
L
= y  y
b
2. Gradients for Second Convolutional Layer Parameters
Before interacting with Conv2, we must backpropagate through the Flatten operation and the
ReLU activation. The vector  (size 8) is unflattened back into a 2 2 2 tensor corresponding
F
to A . Let k  {1, 2} be the channel index, and i, j  {0, 1} be the spatial row and column
2
indices.
L
(k,i,j)
 = =
A2
A
(k,i,j) F 4(k1)+2i+j+1
2
Next, we backpropagate through the ReLU activation (A = max(0, Z )). The derivative of
2 2
ReLU is 1 if the input is strictly greater than 0, and 0 otherwise.
(cid:40)
(k,i,j) (k,i,j)
A 1 if Z > 0
2 = 2
Z (k,i,j) 0 if Z (k,i,j)  0
2 2
(k,i,j)
L A
 (k,i,j) = =  (k,i,j)  2
Z2
Z
(k,i,j) A2
Z
(k,i,j)
2 2
5


The forward pass for Conv2 (stride 1, no padding) computes the feature map Z from the
2
input P :
1
2 2
(k,i,j) (cid:88) (cid:88) (i+m,j+n) (k,m,n) (k)
Z = P K + b
2 1 2 2
m=0 n=0
(k,m,n) (k)
where K is the weight at position (m, n) of the k-th filter, and b is the corresponding
2 2
bias.
Using the multivariable chain rule, the gradient of the loss with respect to a specific filter
(k,m,n)
weight K is the sum of the gradients over all spatial outputs i, j where that weight was
2
applied:
(k,i,j)
Z
2 = P (i+m,j+n)
(k,m,n) 1
K
2
1 1 (k,i,j) 1 1
L = (cid:88) (cid:88) L Z 2 = (cid:88) (cid:88)  (k,i,j) P (i+m,j+n)
K
(k,m,n)
Z
(k,i,j)
K
(k,m,n) Z2 1
2 i=0 j=0 2 2 i=0 j=0
This explicit summation is exactly the definition of a valid 2D cross-correlation (often denoted
(k)
with  in deep learning) between the input feature map P and the error gradient map  .
1 Z2
(k)
Similarly, the gradient with respect to the bias b is the sum of the incoming gradients for
2
that channel:
(k,i,j)
Z
2 = 1
(k)
b
2
1 1 1 1
L (cid:88) (cid:88) L (cid:88) (cid:88) (k,i,j)
=  1 =
b
(k)
Z
(k,i,j) Z2
2 i=0 j=0 2 i=0 j=0
To continue backpropagation down the network, we calculate the gradient with respect to
(x,y)
the input P (for x, y  {0, 1, 2, 3}). The contribution comes from all filters k and all spatial
1
(x,y)
locations (i, j) in Z that utilized P :
2 1
(cid:40)
(k,i,j) (k,xi,yj)
Z K if 0  x  i  2 and 0  y  j  2
2 = 2
P (x,y) 0 otherwise
1
2 1 1 (k,i,j)
 (x,y) = L = (cid:88) (cid:88) (cid:88)  (k,i,j) Z 2
P1
P
(x,y) Z2
P
(x,y)
1 k=1 i=0 j=0 1
Final Result for Second Convolutional Layer Parameters:
1 1
L (k) L (cid:88) (cid:88) (k,i,j) (i+m,j+n)
= P   which mathematically is =  P
K
(k) 1 Z2
K
(k,m,n) Z2 1
2 2 i=0 j=0
1 1
L (cid:88) (cid:88) (k,i,j)
=
b
(k) Z2
2 i=0 j=0
3. Gradients for First Convolutional Layer Parameters
The 2  2 Max Pooling layer with stride 2 produces P by selecting the maximum value from
1
non-overlapping 2  2 windows in A :
1
(x,y) (2x+u,2y+v)
P = max A
1 1
u{0,1},v{0,1}
6


The gradient only flows back to the specific spatial location in A that contained the maximum
1
value. Let r, c  {0, 1, . . . , 7} be the indices in A . The corresponding pooling window indices
1
are x = r/2 and y = c/2.
(cid:40)
(x,y) (r,c) (x,y)
P 1 if A = P
1 = 1 1
A (r,c) 0 otherwise
1
(r/2,c/2)
L P
 (r,c) = =  (r/2,c/2)  1
A1
A
(r,c) P1
A
(r,c)
1 1
We backpropagate through the first ReLU activation (A = max(0, Z )):
1 1
(cid:40)
(r,c) (r,c)
L  if Z > 0
 (r,c) = = A1 1
Z1 Z (r,c) 0 if Z (r,c)  0
1 1
For Conv1, the original input image X (8  8) is zero-padded by 1 to form X of size
pad
10  10. The forward pass is:
2 2
(r,c) (cid:88) (cid:88) (r+m,c+n) (m,n)
Z = X K + b
1 pad 1 1
m=0 n=0
(m,n)
for r, c  {0, 1, . . . , 7}. Here K represents the weights of the single 3  3 filter.
1
Applying the multivariable chain rule, we compute the gradient with respect to the filter
(m,n)
weights K for m, n  {0, 1, 2} by summing over all positions (r, c) where the kernel was
1
applied:
(r,c)
Z
1 = X (r+m,c+n)
(m,n) pad
K
1
7 7 (r,c) 7 7
L = (cid:88) (cid:88) L Z 1 = (cid:88) (cid:88)  (r,c) X (r+m,c+n)
K
(m,n)
Z
(r,c)
K
(m,n) Z1 pad
1 r=0 c=0 1 1 r=0 c=0
Just like in Conv2, this operation is precisely the valid cross-correlation () between the zero-
padded input image X and the local gradient map  .
pad Z1
Finally, the gradient for the bias b is obtained by summing the incoming gradients across
1
the entire feature map Z :
1
(r,c)
Z
1 = 1
b
1
7 7 7 7
L (cid:88) (cid:88) L (cid:88) (cid:88) (r,c)
=  1 =
b
1 Z
(r,c) Z1
r=0 c=0 1 r=0 c=0
Final Result for First Convolutional Layer Parameters:
7 7
L L (cid:88) (cid:88) (r,c) (r+m,c+n)
= X   which mathematically is =  X
K
1
pad Z1
K
(m,n) Z1 pad
1 r=0 c=0
7 7
L (cid:88) (cid:88) (r,c)
=
b
1
Z1
r=0 c=0
7
