function attn_proj_out = attention_token_o_project_step(attn_flat, weights, hyperParameters, cfg)
%ATTENTION_TOKEN_O_PROJECT_STEP Apply output projection to one flattened attention token.

    attn_proj_out = qwen2_runtime.hdl.linear_step(weights.o_proj, attn_flat, cfg);
    if isfield(weights, 'o_bias')
        attn_proj_out = attn_proj_out + weights.o_bias;
    end
end