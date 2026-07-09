# Transformer Encoder


Project 5: Building a Transformer Encoder
Theoretical Concepts & Analysis
prahladha
5.1 Part 1: Scaled Dot-Product Attention
Mathematical Formulation: Given an input sequence representation X  RLd (where L is
the sequence length and d is the embedding dimension), we compute the Query, Key, and Value
matrices:
Q = XW , K = XW , V = XW
Q K V
where W Q , W K  Rdd k and W V  Rddv . The scaled dot-product attention is defined as:
(cid:18) QKT (cid:19)
Attention(Q, K, V ) = softmax  V
d
k
Verification of Attention Weights Summing to 1: Let the attention score matrix before
softmax be S = Q  KT  RLL. The softmax function is applied independently across each row of
d
k
S. For the i-th row and j-th column, the attention weight is:
exp(S )
ij
A =
ij (cid:80)L
exp(S )
k=1 ik
If we sum the weights across any given row i:
(cid:88) L (cid:88) L exp(S ij ) (cid:80)L j=1 exp(S ij )
A = =
ij (cid:80)L
exp(S )
(cid:80)L
exp(S )
j=1 j=1 k=1 ik k=1 ik
Since the index variables j and k both iterate from 1 to L, the numerator and the denominator are
strictly identical. Therefore:
L
(cid:88)
A = 1
ij
j=1
This proves that the attention weights form a valid probability distribution across the sequence for
each token.
5.2 Part 2: Multi-Head Attention
To allow the model to jointly attend to information from different representation subspaces, we
project Q, K, and V into h distinct heads. For each head i  {1, . . . , h}:
(i) (i) (i)
head = Attention(QW , KW , V W )
i Q K V
1


The outputs of all heads are concatenated and passed through a final linear projection matrix W :
O
MultiHead(Q, K, V ) = Concat(head , . . . , head )W
1 h O
5.3 Part 3: Transformer Encoder Block
The Transformer encoder block utilizes a residual architecture with Layer Normalization. Let the
input to the block be X. The forward pass is mathematically defined by two sub-layers:
1. Multi-Head Attention Sub-layer:
X  = LayerNorm(X + MultiHead(X, X, X))
2. Feedforward Sub-layer:
X = LayerNorm(X  + FFN(X  ))
out
Where the Feedforward Network applies two linear transformations with a ReLU activation in
between:
FFN(x) = max(0, xW + b )W + b
1 1 2 2
5.5 Part 5: Training and Optimization (Adam with Bias Correc-
tion)
While Standard Gradient Descent (SGD) updates parameters via  =   g , the Adam
t t1 t
optimizer uses moving averages of the gradient and its square. The mathematical formulation,
emphasizing the bias correction step, is:
m =  m + (1   )g (First moment estimate)
t 1 t1 1 t
v =  v + (1   )g2 (Second moment estimate)
t 2 t1 2 t
Because m and v are initialized to 0, the estimates are biased towards zero during early time
0 0
steps. We apply bias correction:
m
t
m =
t 1  t
1
v
t
v =
t 1  t
2
Final parameter update:

 =    m
t t1 t
v +
t
2


5.6 Part 6: Analysis and Ablations
Ablation B: Positional Encoding Technical focus: Remove positional encoding. Explain the per-
formance change.
Explanation: If positional encodings are removed, the Transformer collapses into a highly
complex, permutation-invariant Bag-of-Words model.
The core operation of the Transformer, self-attention (QKT ), computes similarity scores purely
based on the semantic content of the embeddings, completely agnostic to their sequential position
in the matrix. Without positional embeddings added to the input X, the sequence The movie
was not good, it was bad yields the exact same aggregated representations as The movie was not
bad, it was good.
Consequently, the performance on sentiment classification will drop significantly. The model will
fail to capture structural language dependencies, negations, and syntactic phrasing, as it entirely
loses the concept of word order.
3
