"""Maximum clique as MIS on the complement graph (local addition, see cam/UPSTREAM.md).

As in RLSA (Feng & Yang, 2025, Sec. 4 / App. B), a clique of G is an independent set of the
complement graph G^c, so the energy is the MIS penalty energy of G^c:
    E(x) = -sum_i x_i + beta * sum_{(i,j) not in E, i<j} x_i x_j.
The model keeps message passing on the original graph G (edge_index of the dataset); only the
energy and its gradient use G^c, which is never built: with A the dataset adjacency (edge weight
1/2 per directed edge, as for MIS),
    (A_c x)_i = 1/2 * (sum_{j in graph(i)} x_j - x_i - sum_{j in N(i)} x_j)
             = 1/2 * (S_graph(i) - x_i) - (A x)_i,
which has the same 1/2 scaling as the MIS term, so E and grad = 2*beta*A_c x + b have exactly the
MIS formulas with A replaced by A_c.
"""
import torch
from utils.graph_ops import g_sum


class Env(object):
    def __init__(self, args):
        self.beta = args.beta
        self.solution_type = 'node'
        self.mis_decoding = args.mis_decoding

    @staticmethod
    def complement_matvec(graph, solution):
        """A_c @ solution for the batched complement graph, solution: num_nodes x K"""
        if not hasattr(graph, 'A'):
            graph.A = torch.sparse_coo_tensor(
                graph.edge_index, graph.edge_weight,
                torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
        batch = graph.batch if getattr(graph, 'batch', None) is not None else \
            torch.zeros(graph.num_nodes, dtype=torch.long, device=solution.device)
        per_graph = g_sum(solution, batch)[batch]               # num_nodes x K
        return 0.5*(per_graph - solution) - graph.A@solution

    def energy_func(self, graph, solution, compute_grad=False):
        b = graph.b
        v = self.complement_matvec(graph, solution)
        energy_per_node = solution*(self.beta*v+b)
        batch = graph.batch if getattr(graph, 'batch', None) is not None else \
            torch.zeros(graph.num_nodes, dtype=torch.long, device=solution.device)
        energy = g_sum(energy_per_node, batch)
        grad = 2*self.beta*v+b if compute_grad else None
        return energy, grad

    def _adjacency(self, graph, device):
        """dense boolean adjacency of the (batched, block-diagonal) original graph"""
        n = graph.num_nodes
        adj = torch.zeros(n, n, dtype=torch.bool, device=device)
        adj[graph.edge_index[0], graph.edge_index[1]] = True
        return adj

    def evaluate(self, graph, solution):
        """decode to a clique of G; returns (-clique size, decoded solution) as envs/mis.py"""
        if self.mis_decoding == 'continuous':
            # greedy by probability: add the most likely node still compatible with the clique,
            # then keep only its neighbours in G (same graph), i.e. remove its G^c-neighbours
            x = solution.transpose(0, 1)                             # K x N
            batch = graph.batch if getattr(graph, 'batch', None) is not None else \
                torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device)
            adj = self._adjacency(graph, x.device)
            mask = torch.ones_like(x).bool()
            out = torch.zeros_like(x)
            rows = torch.arange(x.shape[0], device=x.device)
            while mask.any():
                prob = torch.where(mask, x, -torch.inf)
                nxt = torch.argmax(prob, 1)
                live = mask[rows, nxt]
                r, u = rows[live], nxt[live]
                out[r, u] = 1
                mask[r, u] = False
                # nodes of u's graph that are not neighbours of u can no longer join the clique
                drop = (batch.unsqueeze(0) == batch[u].unsqueeze(1)) & ~adj[u]      # len(r) x N
                mask[r] &= ~drop
            return -out.sum(1), out.T
        else:
            # discrete: round, then greedy single flips on the penalty energy until no flip helps
            # (as envs/mis.py); for beta > 1 every local minimum is a valid clique
            solution = torch.round(solution).squeeze(-1)
            batch_idx = torch.arange(solution.shape[1], device=solution.device).long()
            energy, grad = self.energy_func(graph, solution, True)
            best_energy = energy.clone()
            delta = grad*(2*solution-1)/2
            while not (delta <= 0).all():
                value, index = torch.max(delta, dim=0)
                solution[index, batch_idx] = torch.where(value > 0, 1-solution[index, batch_idx],
                                                         solution[index, batch_idx])
                best_energy, grad = self.energy_func(graph, solution, True)
                delta = grad*(2*solution-1)/2
            return best_energy, solution.T
