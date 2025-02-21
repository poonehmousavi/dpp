import torch
import torch.nn as nn
import torch.nn.functional as F

    
class PromptPool(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None, p=0.1, prompt_strategy="sim_strategy", similarity_method ="cosine", residual_threshold=1e-3, attn_threshold=0.05, use_prompt_threshhold=False):
        """
        A prompt pool module that selects the most relevant prompts based on different strategies.

        Args:
            num_prompts (int): Number of available prompts.
            prompt_dim (int): Dimensionality of the prompts.
            model_input_embeds (torch.Tensor, optional): Pretrained prompt values. Defaults to None.
            p (float): Dropout probability.
            strategy (str): Strategy for selecting prompts. Defaults to "sim_strategy".
            residual_threshold (float): Threshold for stopping residual updates in residual_strategy.
            attn_threshold (float): Threshold for discarding low attention prompts in att_strategy.
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_strategy = prompt_strategy  # Strategy to be used for selecting prompts
        self.similarity_method = similarity_method
        self.residual_threshold = residual_threshold
        self.attn_threshold = attn_threshold  # Minimum attention weight for selection
        self.prompt_dim =prompt_dim
        self.use_prompt_threshhold =use_prompt_threshhold

        # Learnable keys
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.01)
        
        # Initialize values
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))
            
        self.dropout = nn.Dropout(p) if p > 0 else None
        self.batch_norm = nn.BatchNorm1d(prompt_dim, affine=True, eps=1e-8)

    
    def sentence_transformer_similarity(self, input_texts):
        """
        Compute similarity using SentenceTransformer model.

        Args:
            input_texts (list of str): List of text inputs.

        Returns:
            torch.Tensor: Similarity scores [B, num_prompts].
            torch.Tensor: Encoded input embeddings.
        """
        with torch.no_grad():
            input_embeddings = self.sentence_transformer.encode(input_texts, convert_to_tensor=True)  # [B, 384]

        # Compute similarities using SentenceTransformer's similarity function
        similarities = self.sentence_transformer.similarity(input_embeddings, self.prompt_keys)  # [B, num_prompts]

        return similarities

    def compute_cosine_similarity(self, input_embedding):
        """Compute cosine similarity between input and prompt keys."""
        norm_input = F.normalize(input_embedding, dim=-1)       # [B, prompt_dim]
        norm_keys = F.normalize(self.prompt_keys, dim=-1)       # [num_prompts, prompt_dim]
        return torch.matmul(norm_input, norm_keys.T)            # [B, num_prompts]

    def sim_strategy(self, input_embedding, top_k=5):
        """
        Similarity-based prompt retrieval strategy.

        Args:
            input_embedding (torch.Tensor): Input embeddings of shape [batch_size, prompt_dim].
            top_k (int): Number of top prompts to select.

        Returns:
            selected_prompts (torch.Tensor): Selected prompt values.
            diversity_loss (torch.Tensor): Diversity loss term.
            topk_indices (torch.Tensor): Indices of selected prompts.
        """
        similarities = self.compute_cosine_similarity(input_embedding)  # [B, num_prompts]
        normalized_similarities = F.softmax(similarities, dim=1)

        if self.dropout is not None:
            normalized_similarities = self.dropout(normalized_similarities)
            normalized_similarities = F.softmax(normalized_similarities, dim=1)

        # Select top-k indices
        topk_values, topk_indices = torch.topk(normalized_similarities, top_k, dim=1)

        # Gather selected prompt values
        selected_prompts = self.prompt_values[topk_indices]  # [B, top_k, prompt_dim]

        # Compute diversity loss
        diversity_loss = - topk_values.sum(dim=1).mean()

        return selected_prompts, diversity_loss, topk_indices

    def residual_strategy(self, input_data, top_k=5):
        """
        Residual-based selection strategy.
        Iteratively selects the most relevant prompt, removes its contribution, and repeats.
        Stops early for individual samples if the residual norm falls below a threshold.

        Args:
            input_embedding (torch.Tensor): Input embeddings of shape [B, prompt_dim].
            top_k (int): Maximum number of prompts to select.

        Returns:
            selected_prompts (torch.Tensor): Selected prompts up to `top_k`, dynamically chosen per sample.
            diversity_loss (torch.Tensor): Encourages diverse prompt selection.
            topk_indices (torch.Tensor): Indices of selected prompts.
        """
        batch_size, _ = input_data.shape
        residual = input_data.clone()  # Copy input embeddings to residual tracker

        selected_prompts = torch.zeros((batch_size, top_k, self.prompt_dim), device=input_data.device)  # [B, top_k, prompt_dim]
        topk_indices = torch.full((batch_size, top_k), -1, device=input_data.device, dtype=torch.long)  # [B, top_k]
        commitment_loss = torch.zeros(batch_size, device=input_data.device)  # Track loss

        active_samples = torch.ones(batch_size, dtype=torch.bool, device=input_data.device)  # Track active samples

        for step in range(top_k):
            if not active_samples.any():  # Stop if all samples have exited early
                break

            # Compute similarities only for active samples
            similarities = self.compute_cosine_similarity(residual)  # [B, num_prompts]
            normalized_similarities = F.softmax(similarities, dim=1)

            # Select top-1 prompt for active samples
            top1_values, top1_indices = torch.topk(normalized_similarities, 1, dim=1)  # [B, 1]
            top1_indices = top1_indices.squeeze(1)  # [B]

            # **Mask-based selection**
            selected_prompts[active_samples, step] = self.prompt_values[top1_indices][active_samples]  # Assign prompts
            topk_indices[active_samples, step] = top1_indices[active_samples]  # Assign indices

            # **Update commitment loss for active samples using scatter_add_()**
            commitment_loss.scatter_add_(0, active_samples.nonzero(as_tuple=True)[0], top1_values.squeeze(1)[active_samples])

            # # **Update residual only for active samples**
            residual = residual - self.prompt_values[top1_indices] * active_samples.unsqueeze(-1)
            
            if self.use_prompt_threshhold:
                residual[active_samples] = self.batch_norm(residual[active_samples])
                # **Check residual norm individually per batch sample**
                residual_norms = torch.norm(residual, dim=-1)  # [B]

                # active_samples &= residual_norms >= self.residual_threshold  # Update active sample mask
                active_samples = active_samples & (residual_norms >= self.residual_threshold)

        # Diversity loss (encourages diverse selections)
        diversity_loss = -commitment_loss.mean()

        return selected_prompts, diversity_loss, topk_indices


    def att_strategy(self, input_embedding, top_k=5):
        """
        Attention-based strategy where the input acts as a query, 
        prompt keys are keys, and prompt values are values.

        Returns:
            selected_prompts: Prompts weighted by attention scores.
            diversity_loss: Entropy-based loss to encourage diverse selections.
            valid_indices: Indices of selected prompts with attention > 0.
        """
        # Compute attention scores (query-key dot product)
        query = input_embedding.unsqueeze(1)  # [B, 1, prompt_dim]
        keys = self.prompt_keys.unsqueeze(0)  # [1, num_prompts, prompt_dim]
        
        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(query, keys.transpose(-1, -2)) / (input_embedding.shape[-1] ** 0.5)  # [B, 1, num_prompts]
        attn_weights = F.softmax(attn_scores, dim=-1).squeeze(1)  # [B, num_prompts]
        
        # Apply dropout if specified
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        if self.use_prompt_threshhold:
            # Zero out attention values below the threshold
            attn_weights = torch.where(attn_weights >= self.attn_threshold, attn_weights, torch.tensor(0.0, device=attn_weights.device))
        
    
        # **Select top-K highest attention prompts**
        topk_attn, topk_idx = torch.topk(attn_weights, k=top_k, dim=-1)  # [B, top_k]
        selected_prompts = self.prompt_values[topk_idx] * topk_attn.unsqueeze(-1)  # [B, top_k, prompt_dim]
        selected_indices = topk_idx  # [B, top_k]

        # Diversity loss: Encourage balanced attention distribution
        diversity_loss = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
        
        return selected_prompts, diversity_loss, selected_indices


    def forward(self, input_data, top_k=5):
        """
        Forward pass to select top-k prompts based on the chosen strategy.

        Args:
            input_data (torch.Tensor): Input embeddings of shape [batch_size, hidden_dimension].
            top_k (int): Number of top prompts to select.

        Returns:
            selected_prompts (torch.Tensor): Selected prompt values.
            diversity_loss (torch.Tensor): Diversity loss term.
            topk_indices (torch.Tensor): Indices of selected prompts.
        """

        if self.prompt_strategy == "sim_strategy":
            return self.sim_strategy(input_data, top_k)
        elif self.prompt_strategy == "residual_strategy":
            return self.residual_strategy(input_data, top_k)
        elif self.prompt_strategy == "att_strategy":
            return self.att_strategy(input_data, top_k)
        else:
            raise NotImplementedError(f"Strategy '{self.prompt_strategy}' is not implemented.")

