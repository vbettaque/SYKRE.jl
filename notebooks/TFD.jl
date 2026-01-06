using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.TFDReplicas

R = 2
N = 1
J = 1
q = 4
β = 30
L = 1000
w = 0.5


syk = SYKData(N, J, q, R, β)

G_init = Replicas.init(R, L)
D2 = Replicas.differentials(R, L; periodic = true)
D2.blocks[L÷2+1, L÷2, 1] *= -1
G_init.blocks[:, :, 1] = inv(D2.blocks[:, :, 1])
G_init.blocks[:, :, 2] *= -1

G, Σ = TFDReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)

Replicas.plot(G; title="q = $(q), beta = $(β)")
