# Scientific formulation

## Storage, sources and losses

For each DU `i`, let `S_i` be the water storage at the end of the preceding time step, and let `I_i` denote conserved water entering the DU. The pre-loss quantity is

\[
A_i(t)=S_i(t-1)+I_i(t).
\]

Prescribed and user-defined sources add water to `I_i`. The `sources` reported by DeepDiscover are additional boundary sources supplied inside a DU. Water entering through the main `water` tensor appears separately.

A physical sink requests a non-negative loss `L_request`. A PET-limited sink uses a non-negative demand `D` and an efficiency `eta` between zero and one:

\[
L_{\rm request}(t)=\eta(t)\min(D(t),A_i(t)).
\]

When the sum of requested losses exceeds available water, all requests are multiplied by the common factor `min(1, A_i / sum(requests))`. After realized losses,

\[
B_i(t)=A_i(t)-\sum_{k}L_{ik}(t).
\]

## CRC processes and conservative allocation

Each process `j` has a non-negative propensity `h_ij` from a neural constitutive relationship. For the standard `CRCNet`, each context variable has an orientation `gamma` in `{+1, 0, -1}`. Network weights are constrained to be non-negative and monotone activations maintain the direct orientation. Positive physical inputs are transformed as `x/(x + x_ref)` before applying signs.

\[
u_{ij}(t)=B_i(t)\frac{h_{ij}(t)}{1+\sum_kh_{ik}(t)},
\qquad
S_{i,raw}(t)=\frac{B_i(t)}{1+\sum_kh_{ik}(t)}.
\]

If capacity `Smax` is finite, retained storage is limited and excess water becomes explicit overflow:

\[
S_i(t)=\min(S_{i,raw}(t),S_{max,i}),\qquad
O_i(t)=\max(S_{i,raw}(t)-S_{max,i},0).
\]

Consequently,

\[
A_i(t)=S_i(t)+\sum_j u_{ij}(t)+O_i(t)+\sum_k L_{ik}(t).
\]

Every process has exactly **one destination**. Destinations refer to another DU or the outlet. The DU list is topologically ordered; all internal routes must point to a DU with a larger index.

## Temporal routing

For route `e`, discharge or transfer released at time `s` is distributed among lags `0...L` by non-negative fractions `g_e,s(tau)` whose sum is one. A routed amount retains the fractions determined at its release time. Water not yet arrived is kept in an explicit transit state. See [routing](routing.md).

## Overall balance

With total DU storage `S_total`, route transit `R_total`, and optional snow storage `S_snow`,

\[
\Delta(S_{total}+R_{total}+S_{snow})
 = I_{external}+I_{sources}-Q_{outlet}-L_{boundaries}+\epsilon.
\]

Numerical conservation and hydrological plausibility are separate evaluations. Persistent storage accumulation may remain physically implausible even when the arithmetic balance closes.

## Informational memory

An optional state `m` can be learned independently of water storage:

\[
m(t)=[1-\lambda(t)]m(t-1)+\lambda(t)z(t).
\]

`m` carries information; it does not contain or move water. Temporal routing uses conservative route kernels; previous discharge is not a recurrent routing state.

## Source material

The equations follow the author's Multiple-DU methods v3 and quantitative Multiple-DU-QQ formalism v3, including capacity, overflow, snow, observation operators, and routing alternatives. The older multi-destination routing search does not define the topology for this version.
