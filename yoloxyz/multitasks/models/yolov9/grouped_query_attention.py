# grouped_query_attention.py
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) module, compatible with nn.MultiheadAttention.
    
    Based on: https://arxiv.org/pdf/2305.13245
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, num_groups=None):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of query attention heads.
            dropout (float): Dropout probability for attention weights.
            batch_first (bool): If True, expects input shape (batch_size, seq_len, embed_dim).
            num_groups (int, optional): Number of key/value groups. Defaults to num_heads // 2.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups if num_groups is not None else max(1, num_heads // 2)
        assert self.num_heads % self.num_groups == 0, "num_heads must be divisible by num_groups"
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        
        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize weights using Xavier uniform and zero biases."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Forward pass for Grouped Query Attention.
        
        Args:
            query: Tensor of shape (seq_len, batch_size, embed_dim) if not batch_first,
                   or (batch_size, seq_len, embed_dim) if batch_first.
            key: Same shape as query.
            value: Same shape as query.
            attn_mask: Optional attention mask.
            key_padding_mask: Optional padding mask of shape (batch_size, seq_len).
            
        Returns:
            output: Tensor of same shape as query.
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        # Handle batch_first input
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        seq_len, batch_size, embed_dim = query.size()
        key_len = key.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(query)  # (seq_len, batch_size, embed_dim)
        k = self.k_proj(key)    # (key_len, batch_size, embed_dim)
        v = self.v_proj(value)  # (key_len, batch_size, embed_dim)
        
        # Reshape for multi-head processing
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        k = k.view(key_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        v = v.view(key_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        # q: (batch_size, num_heads, seq_len, head_dim)
        # k, v: (batch_size, num_heads, key_len, head_dim)
        
        # Group keys and values (as per GQA paper)
        heads_per_group = self.num_heads // self.num_groups
        k_grouped = k.view(batch_size, self.num_groups, heads_per_group, key_len, self.head_dim).contiguous()
        v_grouped = v.view(batch_size, self.num_groups, heads_per_group, key_len, self.head_dim).contiguous()
        k_grouped = k_grouped.mean(dim=2)  # (batch_size, num_groups, key_len, head_dim)
        v_grouped = v_grouped.mean(dim=2)  # (batch_size, num_groups, key_len, head_dim)
        
        # Expand grouped keys and values to match query heads
        # k_grouped = k_grouped.repeat_interleave(heads_per_group, dim=1)
        k_grouped = k_grouped[:, :, None].expand(-1, -1, heads_per_group, -1, -1).reshape(batch_size, self.num_heads, key_len, self.head_dim)
        v_grouped = v_grouped.repeat_interleave(heads_per_group, dim=1)
        # k_grouped, v_grouped: (batch_size, num_heads, key_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k_grouped.transpose(-2, -1)) * self.scale
        # attn_scores: (batch_size, num_heads, seq_len, key_len)
        
        # Apply masks
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, key_len)
            attn_scores = attn_scores.masked_fill(key_padding_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v_grouped)  # (batch_size, num_heads, seq_len, head_dim)
        output = output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, embed_dim)
        output = self.out_proj(output)
        
        # Handle batch_first output
        if self.batch_first:
            output = output.transpose(0, 1)
            attn_weights = attn_weights.transpose(1, 2)  # (batch_size, seq_len, num_heads, key_len)
        
        return output, attn_weights


