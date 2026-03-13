function [attn_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = attention_token_step_sram_handshake_step(start, q_token, k_token, v_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_STEP_SRAM_HANDSHAKE_STEP Multi-cycle external-KV SRAM handshake top.

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
        [~, k_rot] = qwen2_runtime.hdl.attention_rope_single_token_step(q_reg, k_reg, rope_reg, freqs_cis, cfg);
        write_key_reg = reshape(projectedTokenLike(k_rot, cfg), [headDim, numKVHeads]);
        write_value_reg = reshape(projectedTokenLike(v_reg, cfg), [headDim, numKVHeads]);

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
        return;
    end

    if phase == uint8(1)
        read_req = true;
        read_addr = current_read;
        busy = true;
        if read_data_valid
            key_buf(:, :, double(current_read), 1) = reshape(read_key_data, [headDim, numKVHeads]);
            value_buf(:, :, double(current_read), 1) = reshape(read_value_data, [headDim, numKVHeads]);
            if current_read >= total_reads
                phase = uint8(2);
                engine_start_pending = true;
                read_req = false;
            else
                current_read = current_read + uint16(1);
                read_addr = current_read;
            end
        end
    end

    if phase == uint8(2)
        [attn_reg, engine_valid, engine_busy, next_len_reg] = qwen2_runtime.hdl.attention_token_step_sram_engine_step( ...
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
    F = attentionFimath(cfg);
    token = fi(tokenIn, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
end

function F = attentionFimath(cfg)
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', cfg.HDLLinearAccumFractionLength, 'SumMode', 'SpecifyPrecision', 'SumWordLength', cfg.HDLLinearAccumWordLength, 'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end