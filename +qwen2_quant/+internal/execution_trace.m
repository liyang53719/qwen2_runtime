function out = execution_trace(action, varargin)
%EXECUTION_TRACE Collect execution-path decisions during inference.

    persistent trace
    if isempty(trace)
        trace = struct('Op', {}, 'LayerIndex', {}, 'WeightKind', {}, 'ExecKind', {});
    end

    switch lower(action)
        case 'reset'
            trace = struct('Op', {}, 'LayerIndex', {}, 'WeightKind', {}, 'ExecKind', {});
            out = trace;
        case 'log'
            entry = struct('Op', '', 'LayerIndex', -1, 'WeightKind', '', 'ExecKind', '');
            if numel(varargin) >= 1
                entry.Op = char(string(varargin{1}));
            end
            if numel(varargin) >= 2
                entry.LayerIndex = double(varargin{2});
            end
            if numel(varargin) >= 3
                entry.WeightKind = char(string(varargin{3}));
            end
            if numel(varargin) >= 4
                entry.ExecKind = char(string(varargin{4}));
            end
            trace(end+1) = entry; %#ok<AGROW>
            out = entry;
        case 'get'
            out = trace;
        otherwise
            error('execution_trace:UnknownAction', 'Unknown action: %s', action);
    end
end