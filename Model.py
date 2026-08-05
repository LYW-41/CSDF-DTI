import math

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


def scatter_sum(values, index, size):
    output = values.new_zeros((size, values.size(-1)))
    if values.numel() > 0:
        output.index_add_(0, index, values)
    return output


def to_dense_nodes(values, batch_index):
    batch_size = int(batch_index.max().item()) + 1
    counts = torch.bincount(batch_index, minlength=batch_size)
    max_count = int(counts.max().item())
    dense = values.new_zeros((batch_size, max_count, values.size(-1)))
    mask = torch.zeros((batch_size, max_count), dtype=torch.bool, device=values.device)
    for batch_id in range(batch_size):
        selected = values[batch_index == batch_id]
        length = selected.size(0)
        dense[batch_id, :length] = selected
        mask[batch_id, :length] = True
    return dense, mask


def masked_mean(values, mask):
    weight = mask.unsqueeze(-1).to(values.dtype)
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def masked_softmax(scores, mask, dim):
    minimum = torch.finfo(scores.dtype).min
    weights = torch.softmax(scores.masked_fill(~mask, minimum), dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)


def reverse_valid(values, mask):
    output = torch.zeros_like(values)
    lengths = mask.sum(1)
    for index, length in enumerate(lengths.tolist()):
        output[index, :length] = values[index, :length].flip(0)
    return output


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scale):
        ctx.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def gradient_reverse(values, scale=1.0):
    return GradientReversalFunction.apply(values, scale)


class GINEBlock(nn.Module):
    def __init__(self, hidden_dim, bond_dim, dropout):
        super().__init__()
        self.edge_projection = nn.Linear(bond_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, nodes, edge_index, edge_attr):
        source, target = edge_index
        messages = F.relu(nodes[source] + self.edge_projection(edge_attr))
        aggregated = scatter_sum(messages, target, nodes.size(0))
        update = self.mlp((1.0 + self.epsilon) * nodes + aggregated)
        return self.norm(nodes + self.dropout(update))


class DrugEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_projection = nn.Linear(config.atom_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            GINEBlock(config.hidden_dim, config.bond_dim, config.dropout)
            for _ in range(config.gine_layers)
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, graph):
        nodes = self.input_projection(graph.x)
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index, graph.edge_attr)
        return self.output_norm(self.output_projection(nodes))


class FallbackMamba(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim)
        self.in_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = values
        local = self.depthwise(values.transpose(1, 2)).transpose(1, 2)
        content, gate = self.in_projection(local).chunk(2, dim=-1)
        update = self.out_projection(F.silu(content) * torch.sigmoid(gate))
        return self.norm(residual + self.dropout(update))


class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if Mamba is None:
            self.block = FallbackMamba(config.hidden_dim, config.dropout)
        else:
            self.block = Mamba(
                d_model=config.hidden_dim,
                d_state=config.mamba_state_dim,
                d_conv=4,
                expand=2,
            )
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, values):
        if Mamba is None:
            return self.block(values)
        return self.norm(values + self.dropout(self.block(values)))


class BidirectionalMambaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.protein_vocab_size, config.hidden_dim, padding_idx=0)
        self.forward_layers = nn.ModuleList(MambaLayer(config) for _ in range(config.mamba_layers))
        self.backward_layers = nn.ModuleList(MambaLayer(config) for _ in range(config.mamba_layers))
        self.merge = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, protein_ids, protein_mask):
        embedded = self.embedding(protein_ids)
        forward_values = embedded
        for layer in self.forward_layers:
            forward_values = layer(forward_values)
        backward_values = reverse_valid(embedded, protein_mask)
        for layer in self.backward_layers:
            backward_values = layer(backward_values)
        backward_values = reverse_valid(backward_values, protein_mask)
        merged = self.merge(torch.cat([forward_values, backward_values], dim=-1))
        merged = merged * protein_mask.unsqueeze(-1).to(merged.dtype)
        return self.output_norm(self.output_projection(merged))


class PhysicochemicalPriorEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drug_mapping = nn.Sequential(
            nn.Linear(config.atom_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.protein_lookup = nn.Embedding(
            config.protein_vocab_size,
            config.hidden_dim,
            padding_idx=0,
        )
        self.drug_fusion = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.protein_fusion = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.drug_norm = nn.LayerNorm(config.hidden_dim)
        self.protein_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, drug_attributes, protein_ids, drug_hidden, protein_hidden):
        drug_prior = self.drug_mapping(drug_attributes)
        protein_prior = self.protein_lookup(protein_ids)
        drug_fused = self.drug_norm(drug_hidden + self.drug_fusion(drug_prior))
        protein_fused = self.protein_norm(protein_hidden + self.protein_fusion(protein_prior))
        return drug_prior, protein_prior, drug_fused, protein_fused


class CausalShortcutDisentanglement(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_bank = nn.Parameter(torch.randn(config.token_count, config.hidden_dim) * 0.02)
        self.prior_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.token_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.shortcut_classifier = nn.Linear(config.hidden_dim, config.token_count)
        self.causal_classifier = nn.Linear(config.hidden_dim, config.token_count)
        self.temperature = config.token_temperature

    def forward(self, fused, prior, mask):
        projected_prior = F.normalize(self.prior_projection(prior), dim=-1)
        projected_tokens = F.normalize(self.token_projection(self.token_bank), dim=-1)
        scores = torch.einsum("bld,kd->blk", projected_prior, projected_tokens)
        assignment = torch.softmax(scores / self.temperature, dim=-1)
        assignment = assignment * mask.unsqueeze(-1).to(assignment.dtype)
        direction = torch.einsum("blk,kd->bld", assignment, self.token_bank)
        coefficient = (fused * direction).sum(-1, keepdim=True)
        denominator = direction.square().sum(-1, keepdim=True).clamp_min(1e-8)
        shortcut = coefficient / denominator * direction
        shortcut = shortcut * mask.unsqueeze(-1).to(shortcut.dtype)
        causal = (fused - shortcut) * mask.unsqueeze(-1).to(fused.dtype)
        distribution = assignment.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        shortcut_logits = self.shortcut_classifier(masked_mean(shortcut, mask))
        causal_logits = self.causal_classifier(gradient_reverse(masked_mean(causal, mask)))
        return {
            "causal": causal,
            "shortcut": shortcut,
            "distribution": distribution,
            "shortcut_logits": shortcut_logits,
            "causal_logits": causal_logits,
            "assignment": assignment,
        }


class PriorBiasedBidirectionalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        a = config.attention_dim
        p = config.prior_dim
        self.drug_query = nn.Linear(h, a)
        self.drug_key = nn.Linear(h, a)
        self.drug_value = nn.Linear(h, h)
        self.protein_query = nn.Linear(h, a)
        self.protein_key = nn.Linear(h, a)
        self.protein_value = nn.Linear(h, h)
        self.drug_prior = nn.Linear(h, p)
        self.protein_prior = nn.Linear(h, p)
        self.drug_norm = nn.LayerNorm(h)
        self.protein_norm = nn.LayerNorm(h)
        self.prior_bias = config.prior_bias
        self.attention_scale = math.sqrt(a)
        self.prior_scale = math.sqrt(p)

    def forward(self, drug, protein, drug_prior, protein_prior, drug_mask, protein_mask):
        prior = torch.matmul(
            self.drug_prior(drug_prior),
            self.protein_prior(protein_prior).transpose(1, 2),
        ) / self.prior_scale
        pair_mask = drug_mask.unsqueeze(2) & protein_mask.unsqueeze(1)
        drug_scores = torch.matmul(
            self.drug_query(drug),
            self.protein_key(protein).transpose(1, 2),
        ) / self.attention_scale
        drug_attention = masked_softmax(
            drug_scores + self.prior_bias * prior,
            pair_mask,
            dim=2,
        )
        drug_context = torch.matmul(drug_attention, self.protein_value(protein))
        protein_scores = torch.matmul(
            self.protein_query(protein),
            self.drug_key(drug).transpose(1, 2),
        ) / self.attention_scale
        protein_attention = masked_softmax(
            protein_scores + self.prior_bias * prior.transpose(1, 2),
            pair_mask.transpose(1, 2),
            dim=2,
        )
        protein_context = torch.matmul(protein_attention, self.drug_value(drug))
        drug_output = self.drug_norm(drug + drug_context)
        protein_output = self.protein_norm(protein + protein_context)
        joint = torch.cat(
            [masked_mean(drug_output, drug_mask), masked_mean(protein_output, protein_mask)],
            dim=-1,
        )
        return joint, drug_attention, protein_attention


class InteractionPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, joint):
        return self.network(joint).squeeze(-1)


class CSDFDTI(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.drug_encoder = DrugEncoder(self.config)
        self.protein_encoder = BidirectionalMambaEncoder(self.config)
        self.ppe = PhysicochemicalPriorEncoding(self.config)
        self.drug_csd = CausalShortcutDisentanglement(self.config)
        self.protein_csd = CausalShortcutDisentanglement(self.config)
        self.pbf = PriorBiasedBidirectionalFusion(self.config)
        self.predictor = InteractionPredictionHead(self.config)

    def forward(self, batch, return_details=False):
        flat_drug_hidden = self.drug_encoder(batch.graph)
        drug_hidden, drug_mask = to_dense_nodes(flat_drug_hidden, batch.graph.batch)
        drug_attributes, _ = to_dense_nodes(batch.graph.x, batch.graph.batch)
        protein_hidden = self.protein_encoder(batch.protein_ids, batch.protein_mask)
        drug_prior, protein_prior, drug_fused, protein_fused = self.ppe(
            drug_attributes,
            batch.protein_ids,
            drug_hidden,
            protein_hidden,
        )
        drug_split = self.drug_csd(drug_fused, drug_prior, drug_mask)
        protein_split = self.protein_csd(protein_fused, protein_prior, batch.protein_mask)
        joint, drug_attention, protein_attention = self.pbf(
            drug_split["causal"],
            protein_split["causal"],
            drug_prior,
            protein_prior,
            drug_mask,
            batch.protein_mask,
        )
        logits = self.predictor(joint)
        if not return_details:
            return logits
        return logits, {
            "drug_mask": drug_mask,
            "drug_prior": drug_prior,
            "protein_prior": protein_prior,
            "drug_split": drug_split,
            "protein_split": protein_split,
            "drug_attention": drug_attention,
            "protein_attention": protein_attention,
            "joint": joint,
        }

    def loss(self, logits, labels, details):
        classification = F.binary_cross_entropy_with_logits(logits, labels.float())
        auxiliary = logits.new_zeros(())
        for key in ("drug_split", "protein_split"):
            branch = details[key]
            target = branch["distribution"].detach()
            shortcut = -(target * F.log_softmax(branch["shortcut_logits"], dim=-1)).sum(-1).mean()
            adversarial = -(target * F.log_softmax(branch["causal_logits"], dim=-1)).sum(-1).mean()
            auxiliary = auxiliary + shortcut + adversarial
        total = classification + self.config.csd_weight * auxiliary
        return {
            "loss": total,
            "classification_loss": classification,
            "csd_loss": auxiliary,
        }

    def predict_proba(self, batch):
        return torch.sigmoid(self.forward(batch))
