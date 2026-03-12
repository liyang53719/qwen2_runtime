function qwen2_runtime_hdl_attention_multihead_controller_entry_tb
%QWEN2_RUNTIME_HDL_ATTENTION_MULTIHEAD_CONTROLLER_ENTRY_TB MATLAB stimulus for attention core controller baseline.

    cacheLen = 16;
    laneCount = 4;
    numHeads = 2;
    start = false; %#ok<NASGU>
    score_mat = fi(reshape(linspace(-4, -1, cacheLen * numHeads), cacheLen, numHeads), true, 16, 14); %#ok<NASGU>
    value_tensor = fi(reshape(sin(1:(cacheLen * laneCount * numHeads)) / 4, cacheLen, laneCount, numHeads), true, 16, 14); %#ok<NASGU>
    max_seed = fi(-8, true, 16, 14); %#ok<NASGU>
    sum_seed = fi(0, true, 32, 14); %#ok<NASGU>

    clear qwen2_runtime_hdl_attention_multihead_controller_entry

    totalCycles = cacheLen * (3 * laneCount * numHeads) + 16;
    for cyc = 1:totalCycles
        [attn_out, out_valid] = qwen2_runtime_hdl_attention_multihead_controller_entry(cyc == 1, score_mat, value_tensor, max_seed, sum_seed); %#ok<NASGU>
        if out_valid
            break;
        end
    end
    final_attn_out = attn_out; %#ok<NASGU>
end