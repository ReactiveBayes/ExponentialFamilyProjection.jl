using BayesBase

struct BatchLogpdf{F,N}
    batch_logpdf::F
    
    function BatchLogpdf{N}(batch_logpdf::F) where {F,N}
        return new{F,N}(batch_logpdf)
    end
end

# Constructor with batch size
function BatchLogpdf(batch_logpdf::F; batch_size::Int=100) where F
    return BatchLogpdf{batch_size}(batch_logpdf)
end

# Handle batch operations
function (b::BatchLogpdf{F,N})(out::AbstractVector, samples) where {F,N}
    n_samples = length(samples)
    n_batches = ceil(Int, n_samples / N)
    
    # Process samples in batches
    for i in 1:n_batches
        start_idx = (i-1) * N + 1
        end_idx = min(i * N, n_samples)
        batch_slice = start_idx:end_idx
        
        # Process current batch
        view(out, batch_slice).= b.batch_logpdf(view(samples, batch_slice))
    end
    
    return out
end

# Handle single sample operations (vector input)
function (b::BatchLogpdf{F,N})(x::AbstractVector) where {F,N}
    out = zeros(length(x))
    b(out, x)  # Use the batch operation method
    return out
end

function (b::BatchLogpdf)(x::Real)
    return b.batch_logpdf(x)
end

# Convert regular logpdf function to BatchLogpdf
function Base.convert(::Type{BatchLogpdf}, logpdf_fn)
    return BatchLogpdf{100}(logpdf_fn)  # Default batch size
end

function Base.convert(::Type{BatchLogpdf{N}}, logpdf_fn) where N
    return BatchLogpdf{N}(logpdf_fn)
end

function Base.convert(::Type{BatchLogpdf}, logpdf_fn::BatchLogpdf{F,N}) where {F,N}
    return logpdf_fn
end   