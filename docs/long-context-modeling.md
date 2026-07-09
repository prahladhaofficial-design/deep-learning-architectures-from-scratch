# Long-Context Modeling


6.Long-Context Modeling
Theoretical Solutions
prahladha
6.1 Part 1: Computational Scaling
1. Time complexity per layer of a Transformer encoder: A standard Transformer encoder
layer consists of Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN). Let L be
the sequence length and d be the hidden dimension.
 Self-Attention: Computing the query, key, and value matrices takes O(Ld2). Multiplying
Q and KT takes O(L2d). Multiplying the attention scores by V takes O(L2d). Finally, the
output linear projection takes O(Ld2). Total for attention: O(L2d + Ld2).
 FFN: The two linear layers in the FFN expand the dimension (usually to 4d) and project it
back to d. This takes O(Ld2).
Therefore, the total time complexity per layer of a Transformer encoder is:
O(L2d + Ld2)
2. Per-sequence time complexity of a standard RNN: For a standard RNN processing a
sequence of length L, at each time step t, the hidden state is updated as h = f (W h +W x +b).
t h t1 x t
 Multiplying W h takes O(d2).
h t1
 Multiplying W x takes O(d2).
x t
Since this operation must be performed sequentially L times, the per-sequence time complexity is:
O(Ld2)
3. Ratio of Transformer cost to RNN cost: Given L = 1024 and d = 256, we compute the
ratio of their theoretical time complexities (ignoring constant factors for the asymptotic scaling):
L2d + Ld2 L
Ratio = = + 1
Ld2 d
Substitute the given values:
1024
Ratio = + 1 = 4 + 1 = 5
256
(Note: If only the self-attention mechanism O(L2d) is compared to the RNN O(Ld2) strictly ignor-
ing the projection weights, the ratio is simply L = 4.)
d
1


6.2 Part 2: Gradient Propagation in RNNs
Consider the recurrence: h = Ah + Bx .
t t1 t
1. Express h in terms of h and {x , . . . , x }: By unrolling the recurrence relation:
t 0 1 t
h = Ah + Bx
1 0 1
h = A(Ah + Bx ) + Bx = A2h + ABx + Bx
2 0 1 2 0 1 2
.
.
.
h = Ath + At1Bx + At2Bx +    + A0Bx
t 0 1 2 t
This can be written compactly as:
t
(cid:88)
h = Ath + AtiBx
t 0 i
i=1
2. Show that ht = Ak: From the recurrence, we can express h directly in terms of an
h t
tk
intermediate state h :
tk
t
(cid:88)
h = Akh + AtiBx
t tk i
i=tk+1
Taking the partial derivative of h with respect to h , the summation term drops out because it
t tk
does not depend on h . Thus:
tk
h
t = Ak
h
tk
3. Spectral radius (A) implications: Let (A) be the spectral radius (the maximum absolute
eigenvalue) of matrix A.
 If (A) > 1: As k grows, the matrix power Ak grows exponentially. This leads to the
exploding gradient problem, making training highly unstable.
 If (A) < 1: As k grows, Ak decays exponentially toward zero. This leads to the vanishing
gradient problem, making the model unable to learn long-term dependencies.
6.3 Part 3: State Space Models (SSM)
Consider the discrete SSM: h = Ah + Bx and y = Ch .
k k1 k k k
1. Show the equation for y : Assuming the initial hidden state before the sequence starts is
k
h = 0, we can unroll the state equation from i = 0 to k:
1
k
(cid:88)
h = AkiBx
k i
i=0
2


Substituting h into the observation equation y = Ch :
k k k
(cid:32) k (cid:33)
(cid:88)
y = C AkiBx
k i
i=0
Because matrix multiplication is distributive, we can bring C inside the sum:
k
(cid:88)
y = CAkiBx
k i
i=0
2. Interpret this as a 1D convolution: A standard discrete 1D convolution is defined as
(K  x) =
(cid:80)k
K x . By defining the kernel weight at step j as:
k i=0 ki i
K = CAjB
j
We can rewrite our equation for y as:
k
k
(cid:88)
y = K x
k ki i
i=0
This perfectly matches the formulation of a causal 1D convolution with kernel K.
3. Time-invariance and one limitation:
 Why it is time-invariant: The system is time-invariant because the transition matrices A,
B, and C are static; they do not change with the time step k. Consequently, the convolutional
kernel K is fixed, meaning a shift in the input sequence strictly results in an identical shift
in the output sequence.
 Limitation: Because the kernel is fixed and data-independent, the model cannot perform
content-based reasoning. It applies the exact same filtering to the sequence regardless of what
the actual input values (x ) are, making it unable to selectively focus on or forget specific
k
tokens based on their context (unlike attention mechanisms).
6.4 Part 4: Selective State Space Models
Selective SSM introduces: h = Ah + Bx , y = g  (Ch ), and g = (W x ).
k k1 k k k k k g k
1. Verify that all dimensions are consistent: Let the input dimension be d and the hidden
in
state dimension be d .
hidden
 x
k
 Rdin
 h k  Rd hidden
 A  Rd hidden d hidden
 B  Rd hidden din
3


 C  Rd hidden d hidden (as per the Python implementation prompt)
The output of Ch
k
will be a vector in Rd hidden. For the element-wise multiplication (Hadamard
product ) to be valid, the gate vector g must also be of dimension d . Since g = (W x ),
k hidden k g k
the weight matrix W must map from d to d . Therefore:
g in hidden
W  Rd hidden din
g
With these dimensions, g k  Rd hidden and Ch k  Rd hidden, making the operation y k = g k  (Ch k )
perfectly dimensionally consistent, resulting in y
k
 Rd hidden.
2. Explain why this model is no longer time-invariant: The gate g = (W x ) depends
k g k
explicitly on the input x at time step k. Because x changes at every time step, the gating vector
k k
g also changes at every time step. The transformation applied to the hidden state is now dependent
k
on the data at that specific moment in time. Thus, the systems response to an input is no longer
independent of when that input occurs; it varies dynamically based on the input sequence content.
3. Show the equation for y and why it cannot be a fixed convolution: From Part 3, we
k
know that Ch = (cid:80)k CAkiBx . Substituting this into the new observation equation:
k i=0 i
(cid:32) k (cid:33)
(cid:88)
y = g  (Ch ) = g  CAkiBx
k k k k i
i=0
Because the element-wise multiplication g depends on the current input x , the effective kernel
k k
applied to the past inputs x is now g  (CAkiB). This cannot be represented as a fixed convo-
i k
lution because a standard convolution requires a static, data-independent kernel matrix K. Here,
the kernel dynamically changes at every single time step k due to the presence of g , preventing
k
it from being computed purely via a fast Fourier transform (FFT) or standard fixed convolutional
filters.
4
