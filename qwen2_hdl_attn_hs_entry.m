function [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = qwen2_hdl_attn_hs_entry(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg)
%QWEN2_HDL_ATTN_HS_ENTRY Short-name HDL entry for handshake attention controller.

    hiddenSize = 1536;
    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;

    persistent q_token_reg k_token_reg v_token_reg
    if isempty(q_token_reg)
        q_token_reg = initTokenBuffer(headDim, numHeads, cfg, h_token);
        k_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
        v_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
    end

    attn_proj_out = initVectorBuffer(hiddenSize, cfg, h_token);
    if start
        [q_token_reg, k_token_reg, v_token_reg] = qwen2_runtime.hdl.attention_token_qkv_project_step(h_token, weights, hyperParameters, cfg);
    end

    [attn_flat, attn_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        attention_token_step_sram_handshake_step_local( ...
            start, q_token_reg, k_token_reg, v_token_reg, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg);

    out_valid = false;
    if attn_valid
        attn_proj_out = qwen2_runtime.hdl.attention_token_o_project_step(attn_flat, weights, hyperParameters, cfg);
        out_valid = true;
    end
end

function [attn_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = attention_token_step_sram_handshake_step_local(start, q_token, k_token, v_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_STEP_SRAM_HANDSHAKE_STEP_LOCAL Localized handshake step to bypass package-only wrapper boundaries.

    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;
    maxCacheLen = 256;

    persistent phase current_read total_reads q_reg k_reg v_reg rope_reg key_buf value_buf attn_reg valid_reg next_len_reg write_addr_reg shift_reg write_key_reg write_value_reg engine_start_pending
    if isempty(phase)
        F = attentionFimath(cfg);
        phase = uint8(0);
        current_read = uint16(1);
        total_reads = uint16(0);
        q_reg = fi(zeros(headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        k_reg = fi(zeros(headDim, numKVHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        v_reg = fi(zeros(headDim, numKVHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        rope_reg = uint16(1);
        key_buf = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_buf = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        attn_reg = fi(zeros(headDim * numHeads, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        valid_reg = false;
        next_len_reg = uint16(0);
        write_addr_reg = uint16(1);
        shift_reg = false;
        write_key_reg = fi(zeros(headDim, numKVHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        write_value_reg = fi(zeros(headDim, numKVHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        engine_start_pending = false;
    end

    attn_out = attn_reg;
    out_valid = valid_reg;
    busy = (phase ~= uint8(0));
    read_req = false;
    read_addr = current_read;
    write_req = false;
    write_addr = write_addr_reg;
    shift_enable = shift_reg;
    write_key_token = write_key_reg;
    write_value_token = write_value_reg;
    next_valid_len = next_len_reg;
    valid_reg = false;

    if start
        q_reg = projectedTokenLike(q_token, cfg);
        k_reg = projectedTokenLike(k_token, cfg);
        v_reg = projectedTokenLike(v_token, cfg);
        rope_reg = cast(rope_position, 'like', uint16(0));
        key_buf(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, attentionFimath(cfg));
        value_buf(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, attentionFimath(cfg));
        total_reads = cast(min(double(cache_valid_len), maxCacheLen), 'like', uint16(0));
        current_read = uint16(1);
        shift_reg = double(cache_valid_len) >= maxCacheLen;
        if shift_reg
            write_addr_reg = uint16(maxCacheLen);
        else
            write_addr_reg = uint16(double(cache_valid_len) + 1);
        end
        [~, k_rot] = attention_rope_single_token_step_local(q_reg, k_reg, rope_reg, freqs_cis, cfg);
        write_key_reg = projectedTokenLike(k_rot, cfg);
        write_value_reg = projectedTokenLike(v_reg, cfg);

        if total_reads == uint16(0)
            phase = uint8(2);
            engine_start_pending = true;
            busy = true;
        else
            phase = uint8(1);
            busy = true;
            read_req = true;
            read_addr = current_read;
        end
        write_addr = write_addr_reg;
        shift_enable = shift_reg;
        write_key_token = write_key_reg;
        write_value_token = write_value_reg;
    elseif phase == uint8(1)
        read_req = true;
        read_addr = current_read;
        busy = true;
        if read_data_valid
            key_buf(:, :, double(current_read), 1) = read_key_data;
            value_buf(:, :, double(current_read), 1) = read_value_data;
            if current_read >= total_reads
                phase = uint8(2);
                engine_start_pending = true;
                read_req = false;
            else
                current_read = current_read + uint16(1);
                read_addr = current_read;
            end
        end
    elseif phase == uint8(2)
        [attn_reg, engine_valid, engine_busy, next_len_reg] = attention_token_step_sram_engine_step_local( ...
            engine_start_pending, q_reg, k_reg, v_reg, key_buf, value_buf, total_reads, rope_reg, freqs_cis, hyperParameters, cfg);
        engine_start_pending = false;
        attn_out = attn_reg;
        next_valid_len = next_len_reg;
        busy = engine_busy;
        if engine_valid
            out_valid = true;
            write_req = true;
            phase = uint8(0);
            busy = false;
        end
    end
end

function token = projectedTokenLike(tokenIn, cfg)
    token = fi(tokenIn, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, attentionFimath(cfg));
end

function [xq_rot, xk_rot] = attention_rope_single_token_step_local(xq, xk, rope_position, freqs_cis, cfg)
    F = localFimath(cfg);
    pos = double(rope_position);
    cosTheta = fi(freqs_cis.Cos(:, pos), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    sinTheta = fi(freqs_cis.Sin(:, pos), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    xq_rot = rotateTokenPairLocal(xq, cosTheta, sinTheta, F, cfg);
    xk_rot = rotateTokenPairLocal(xk, cosTheta, sinTheta, F, cfg);
end

function x_rot = rotateTokenPairLocal(x, cosTheta, sinTheta, F, cfg)
    half = size(x, 1) / 2;
    headCount = size(x, 2);
    x_rot = fi(zeros(size(x)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    for h = 1:headCount
        realPart = fi(x(1:half, h), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        imagPart = fi(x(half+1:end, h), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        x_rot(1:half, h) = fi(realPart .* cosTheta - imagPart .* sinTheta, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        x_rot(half+1:end, h) = fi(realPart .* sinTheta + imagPart .* cosTheta, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    end
end

function [attn_out, out_valid, busy, next_valid_len] = attention_token_step_sram_engine_step_local(start, q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, ~, cfg)
    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;
    maxCacheLen = 256;
    F = attentionFimath(cfg);
    persistent running head_idx q_rot_reg key_cache_reg value_cache_reg next_len_reg attn_reg active_len_reg scale_reg head_start_pending
    if isempty(running)
        running = false;
        head_idx = uint8(1);
        q_rot_reg = fi(zeros(headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        key_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        next_len_reg = uint16(0);
        attn_reg = fi(zeros(headDim * numHeads, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        active_len_reg = uint16(0);
        scale_reg = fi(headDimScaleLocal(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        head_start_pending = false;
    end

    attn_out = attn_reg;
    out_valid = false;
    busy = running;
    next_valid_len = next_len_reg;

    if start
        q_fix = projectedTokenLike(q_token, cfg);
        k_fix = projectedTokenLike(k_token, cfg);
        v_fix = projectedTokenLike(v_token, cfg);
        [q_rot_reg, k_rot] = attention_rope_single_token_step_local(q_fix, k_fix, rope_position, freqs_cis, cfg);

        [key_cache_reg, value_cache_reg, next_valid_len_raw] = kv_cache_update_step_local( ...
            key_cache_in, value_cache_in, double(cache_valid_len), k_rot, v_fix);

        next_len_reg = cast(next_valid_len_raw, 'like', uint16(0));
        next_valid_len = next_len_reg;
        active_len_reg = uint16(min(double(next_valid_len_raw), maxCacheLen));
        attn_reg(:) = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        running = true;
        head_idx = uint8(1);
        scale_reg = fi(headDimScaleLocal(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        head_start_pending = true;
    end

    if ~running
        return;
    end

    kv_head_idx = kvHeadIndexLocal(head_idx, numHeads, numKVHeads);
    scale_fix = fi(scale_reg, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    query_vec = fi(q_rot_reg(:, double(head_idx)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    key_mat = cacheHeadSliceToMatrixLocal(key_cache_reg, kv_head_idx, headDim, maxCacheLen, cfg);
    value_mat = cacheHeadSliceToMatrixLocal(value_cache_reg, kv_head_idx, headDim, maxCacheLen, cfg);
    [head_acc, head_valid] = attention_token_step_sram_single_head_step_local( ...
        head_start_pending, query_vec, key_mat, value_mat, active_len_reg, scale_fix, cfg);
    head_start_pending = false;
    if head_valid
        flat_idx = (double(head_idx) - 1) * headDim + 1;
        attn_reg(flat_idx:flat_idx + headDim - 1) = fi(head_acc, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        if head_idx >= uint8(numHeads)
            attn_out = attn_reg;
            out_valid = true;
            running = false;
        else
            head_idx = head_idx + uint8(1);
            head_start_pending = true;
        end
    end

    busy = running;
    next_valid_len = next_len_reg;
end

function [key_cache_out, value_cache_out, next_valid_len] = kv_cache_update_step_local(key_cache_in, value_cache_in, cache_valid_len, key_token, value_token)
    key_cache_out = fi(key_cache_in, true, 16, 14);
    value_cache_out = fi(value_cache_in, true, 16, 14);
    key_token = fi(key_token, true, 16, 14);
    value_token = fi(value_token, true, 16, 14);
    maxLenIndex = size(key_cache_in, 3);
    maxLen = cast(maxLenIndex, 'like', cache_valid_len);
    next_valid_len = cache_valid_len;

    if cache_valid_len < maxLen
        insertIdx = cache_valid_len + cast(1, 'like', cache_valid_len);
        for d = 1:size(key_cache_in, 1)
            for h = 1:size(key_cache_in, 2)
                key_cache_out(d, h, double(insertIdx), 1) = key_token(d, h);
                value_cache_out(d, h, double(insertIdx), 1) = value_token(d, h);
            end
        end
        next_valid_len = insertIdx;
    else
        for pos = 1:maxLenIndex-1
            for d = 1:size(key_cache_in, 1)
                for h = 1:size(key_cache_in, 2)
                    key_cache_out(d, h, pos, 1) = key_cache_in(d, h, pos + 1, 1);
                    value_cache_out(d, h, pos, 1) = value_cache_in(d, h, pos + 1, 1);
                end
            end
        end
        for d = 1:size(key_cache_in, 1)
            for h = 1:size(key_cache_in, 2)
                key_cache_out(d, h, maxLenIndex, 1) = key_token(d, h);
                value_cache_out(d, h, maxLenIndex, 1) = value_token(d, h);
            end
        end
        next_valid_len = maxLen;
    end
end

function [head_acc_out, head_valid] = attention_token_step_sram_single_head_step_local(start, query_vec, key_mat, value_mat, active_len, scale_fix, cfg)
    maxCacheLen = size(key_mat, 2);
    Fmax = maxPathFimathLocal();
    Fsum = sumPathFimathLocal();
    persistent phase running max_reg sum_reg recip_reg score_cache_reg exp_cache_reg weight_cache_reg phase_start_pending
    if isempty(phase)
        phase = uint8(0);
        running = false;
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        score_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        exp_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        weight_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        phase_start_pending = false;
    end

    head_acc_out = fi(zeros(size(value_mat, 1), 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, attentionFimath(cfg));
    head_valid = false;

    if start
        phase = uint8(1);
        running = true;
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        score_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        exp_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        weight_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        phase_start_pending = true;
    end

    if ~running
        return;
    end

    if phase == uint8(1)
        [score_cache_reg, fill_valid] = qwen2_runtime.hdl.attention_token_step_sram_score_cache_fill_step( ...
            phase_start_pending, query_vec, key_mat, active_len, scale_fix);
        phase_start_pending = false;
        if fill_valid
            phase = uint8(2);
            phase_start_pending = true;
        end
    elseif phase == uint8(2)
        max_seed = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        [max_reg, max_valid] = qwen2_runtime.hdl.attention_token_step_sram_max_scan_step( ...
            phase_start_pending, score_cache_reg, active_len, max_seed);
        phase_start_pending = false;
        if max_valid
            phase = uint8(3);
            phase_start_pending = true;
        end
    elseif phase == uint8(3)
        sum_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        [sum_reg, recip_reg, exp_cache_reg, sum_valid] = qwen2_runtime.hdl.attention_token_step_sram_sum_scan_step( ...
            phase_start_pending, score_cache_reg, active_len, max_reg, sum_seed);
        phase_start_pending = false;
        if sum_valid
            weight_cache_reg = qwen2_runtime.hdl.attention_token_step_sram_weight_finalize_step(exp_cache_reg, recip_reg, active_len);
            phase = uint8(4);
            phase_start_pending = true;
        end
    else
        value_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, valuePathFimathLocal());
        [head_acc_out, head_done] = qwen2_runtime.hdl.attention_token_step_sram_head_value_scan_step( ...
            phase_start_pending, weight_cache_reg, value_mat, value_seed, cfg);
        phase_start_pending = false;
        if head_done
            running = false;
            phase = uint8(0);
            head_valid = true;
        end
    end
end

function matrix = cacheHeadSliceToMatrixLocal(cacheTensor, headIndex, headDim, maxCacheLen, cfg)
    F = attentionFimath(cfg);
    matrix = fi(zeros(headDim, maxCacheLen), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    for pos = 1:maxCacheLen
        for dim = 1:headDim
            matrix(dim, pos) = cacheTensor(dim, headIndex, pos, 1);
        end
    end
end

function idx = kvHeadIndexLocal(headIdx, numHeads, numKVHeads)
    if numHeads == 12 && numKVHeads == 2
        if headIdx <= uint8(6)
            idx = 1;
        else
            idx = 2;
        end
    else
        headsPerKV = uint8(numHeads / numKVHeads);
        if headIdx <= headsPerKV
            idx = 1;
        else
            idx = numKVHeads;
        end
    end
end

function scale = headDimScaleLocal(headDim, cfg)
    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        switch double(headDim)
            case 128
                scale = single(0.0883883476483184);
            case 64
                scale = single(0.125);
            otherwise
                scale = single(1.0 / sqrt(double(headDim)));
        end
    else
        scale = single(1.0 / sqrt(double(headDim)));
    end
end

function F = maxPathFimathLocal()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 16, ...
        'SumFractionLength', 14);
end

function F = sumPathFimathLocal()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end

function F = valuePathFimathLocal()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end

function value = initTokenBuffer(headDim, headCount, cfg, prototype)
    if isFixedPointMode(cfg)
        F = controllerFimath(cfg);
        if isa(prototype, 'embedded.fi')
            value = fi(zeros(headDim, headCount), true, prototype.WordLength, prototype.FractionLength, F);
        else
            value = fi(zeros(headDim, headCount), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        end
    else
        value = zeros(headDim, headCount, 'single');
    end
end

function value = initVectorBuffer(hiddenSize, cfg, ~)
    if isFixedPointMode(cfg)
        value = fi(zeros(hiddenSize, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, controllerFimath(cfg));
    else
        value = zeros(hiddenSize, 1, 'single');
    end
end

function F = controllerFimath(cfg)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', cfg.HDLLinearAccumFractionLength, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', cfg.HDLLinearAccumWordLength, ...
        'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end

function F = attentionFimath(cfg)
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', cfg.HDLLinearAccumFractionLength, 'SumMode', 'SpecifyPrecision', 'SumWordLength', cfg.HDLLinearAccumWordLength, 'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end

function F = localFimath(cfg)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', cfg.HDLLinearAccumFractionLength, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', cfg.HDLLinearAccumWordLength, ...
        'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end

function tf = isFixedPointMode(cfg)
    tf = isstruct(cfg) && logical(cfg.UseFixedPointHDL);
end