include("A2_ResNet50.jl")
using Lux
function ModelBackbone(; sample_length::Int)
    model_embedding = Chain(ModelResNet50(),
                            GlobalMeanPool(),
                            FlattenLayer(),
                            Dense(2048 => sample_length),
                            Dense(sample_length => 4*sample_length, relu),
                            Dense(4*sample_length => 4*sample_length, relu),
                            Dense(4*sample_length => 1))
    return model_embedding
end