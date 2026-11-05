# Scientific formulation

## Storage, sources and losses

For each DiscoverUnit $i$, let $S_i(t-1)$ denote the storage at the end of the previous time step and $I_i(t)$ the incoming water. The water available before boundary losses is

$$
A_i(t)=S_i(t-1)+I_i(t).
$$

Prescribed and user-defined sources add water to $I_i(t)$. The `sources` reported by DeepDiscover are additional boundary sources supplied inside a DU. Water entering through the main `water` tensor appears separately.

A physical sink requests a non-negative loss. A demand-limited sink uses a non-negative demand $D(t)$ and an efficiency $\eta(t)$ between zero and one:

$$
L_{\mathrm{request}}(t)=\eta(t)\min(D(t),A_i(t)).
$$

If the combined requested losses exceed the available water, each request is scaled by the same factor, $\min\left(1, A_i / \sum_k L_{ik,\mathrm{request}}\right)$. After the realized losses,

$$
B_i(t)=A_i(t)-\sum_{k}L_{ik}(t).
$$

## CRC processes and conservative allocation

Each process $j$ has a non-negative propensity $h_{ij}(t)$ determined by a learned constitutive relationship. In `CRCNet`, each context variable has a CRC orientation $\gamma\in\{-1,0,+1\}$. Non-negative network weights and monotone activations enforce the specified direct orientation. Non-negative input variables are scaled using $x/(x+x_{\mathrm{ref}})$.

$$
u_{ij}(t)=B_i(t)\frac{h_{ij}(t)}{1+\sum_kh_{ik}(t)},
\qquad
S_{i,\mathrm{raw}}(t)=\frac{B_i(t)}{1+\sum_kh_{ik}(t)}.
$$

For a finite capacity $S_{\max,i}$, excess retained water becomes overflow:

$$
S_i(t)=\min(S_{i,\mathrm{raw}}(t),S_{\max,i}),\qquad
O_i(t)=\max(S_{i,\mathrm{raw}}(t)-S_{\max,i},0).
$$

Consequently,

$$
A_i(t)=S_i(t)+\sum_j u_{ij}(t)+O_i(t)+\sum_k L_{ik}(t).
$$

Every process has exactly **one destination**. Destinations refer to another DU or the outlet. The DU list is topologically ordered; all internal routes must point to a DU with a larger index.

## Temporal routing

For a route $e$, water released at time $s$ is distributed across lags $\tau=0,\ldots,L_e$ using a routing kernel $g_{e,s}(\tau)$ such that

$$
 g_{e,s}(\tau)\geq 0,\qquad
 \sum_{\tau=0}^{L_e} g_{e,s}(\tau)=1.
$$

Each released quantity retains the routing weights calculated at its release time. Water awaiting arrival remains in the transit state. See [Temporal routing](routing.md).

## Overall balance

The complete balance accounts for water stored in the DUs, water in transit and, when enabled, snow storage:

$$
\Delta(S_{\mathrm{total}}+R_{\mathrm{total}}+S_{\mathrm{snow}})
 = I_{\mathrm{external}}+I_{\mathrm{sources}}-Q_{\mathrm{outlet}}-L_{\mathrm{boundaries}}+\epsilon.
$$

Numerical conservation and hydrological plausibility are separate evaluations. Persistent storage accumulation may remain physically implausible even when the arithmetic balance closes.

## Informational memory

The optional informational memory $m(t)$ evolves independently of water storage:

$$
m(t)=[1-\lambda(t)]m(t-1)+\lambda(t)z(t).
$$

The memory state carries information and does not transport water. All water transferred between DiscoverUnits is accounted for through the conservative routing kernels.
