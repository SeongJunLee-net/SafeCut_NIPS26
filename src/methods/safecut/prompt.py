import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, ctx_init: str = "a photo of a"):
        super().__init__()
        dtype  = torch.float32
        device = clip_model.token_embedding.weight.device
        ctx_init = ctx_init.replace("_", " ")
        prompt   = clip.tokenize(ctx_init).to(device)  
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)  
        n_ctx       = len(ctx_init.split())  
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]   
        self.ctx    = nn.Parameter(ctx_vectors)
        classnames        = [name.replace("_", " ") for name in classnames]
        prompts           = [ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        ).to(device)  
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  
        self.register_buffer("token_prefix",      embedding[:, :1, :])  
        self.register_buffer("token_suffix",      embedding[:, 1 + n_ctx:, :])  
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
    def forward(self) -> torch.Tensor:
        ctx = self.ctx  
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)  
def encode_text_with_prompt(clip_model, prompt_learner: PromptLearner) -> torch.Tensor:
    prompts           = prompt_learner()  
    tokenized_prompts = prompt_learner.tokenized_prompts  
    x = prompts + clip_model.positional_embedding.type(prompts.dtype)  
    x = x.permute(1, 0, 2)                        
    x = clip_model.transformer(x)  
    x = x.permute(1, 0, 2)                        
    x = clip_model.ln_final(x).type(prompts.dtype)  
    x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]  
    x = x @ clip_model.text_projection  
    return x  