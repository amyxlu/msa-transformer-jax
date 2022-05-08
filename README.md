# msa-transformer-jax
This is a JAX implementation of MSA Transformer. Starting from a standard Transformer codebase from Flax examples, new modules are defined.

The key is that the underlying modules for attention are different. To implement row-attention rather than column attention, a simple einsum trick will do. That is, in row attention you would have an attention map with shape  `(..., n_rows, n_rows)`, in column attention you would have an attention map with shape `(..., n_cols, n_cols)`. This requires only a simple change to the function used to calculate the attention matrix, and in fact, the heavy lifting is done using einsum. The additional dimension arising from the `n_sequences_per_msa` is also automatically taken care of by einsum.

After linear projection, the query, key, and value matrices have shape `(N, M, L, D_kv, D_kv, D_emb / H_heads)`
`[N_batches, M_additional_dim, L_qkv, H_heads, D_features_per_head]`

For row attention, which attends to different positions in the sequence (i.e. recapitulates the contact map), the dimensions refer to:
`(N_msas, M_num_rows, L_num_cols, H_heads, D_features_per_head)`.

For column attention, which attends to different species across the MSA, the matrices have shape `(N_msas, M_num_cols, L_num_rows, H_heads, D_features_per_head)`.

That is, the same matrix self-attention implementation can be used for both row, 
