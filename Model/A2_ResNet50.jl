include("A1_ResNet50Block.jl")
using Lux
function ModelStage0()
    model_stage0 = @compact(w1 = Conv((7,7), 3=>64; stride=2, pad=3),
                            w2 = BatchNorm(64, relu),
                            w3 = MaxPool((3,3); stride=2, pad=1)) do x
                        y1 = w1(x)
                        y2 = w2(y1)
                        out = w3(y2)
                    @return out
                    end
    return model_stage0
end
function ModelStage1()
    model_stage1 = @compact(w1 = ModelBtnk1(64, 64, 1),
                            w2 = ModelBtnk2(256),
                            w3 = ModelBtnk2(256)) do x
                        y1 = w1(x)
                        y2 = w2(y1)
                        out = w3(y2)       
                   @return out
                   end
    return model_stage1
end
function ModelStage2()
    model_stage2 = @compact(w1 = ModelBtnk1(256, 128, 2),
                            w2 = ModelBtnk2(512),
                            w3 = ModelBtnk2(512),
                            w4 = ModelBtnk2(512)) do x
                        y1 = w1(x)
                        y2 = w2(y1)
                        y3 = w3(y2)
                        out = w4(y3)
                   @return out
                   end
    return model_stage2
end
function ModelStage3()
    model_stage3 = @compact(w1 = ModelBtnk1(512, 256, 2),
                            w2 = ModelBtnk2(1024),
                            w3 = ModelBtnk2(1024),
                            w4 = ModelBtnk2(1024),
                            w5 = ModelBtnk2(1024),
                            w6 = ModelBtnk2(1024)) do x
                        y1 = w1(x)
                        y2 = w2(y1)
                        y3 = w3(y2)
                        y4 = w4(y3)
                        y5 = w5(y4)
                        out = w6(y5)
                   @return out
                   end
    return model_stage3
end
function ModelStage4()
    model_stage4 = @compact(w1 = ModelBtnk1(1024, 512, 2),
                            w2 = ModelBtnk2(2048),
                            w3 = ModelBtnk2(2048)) do x
                        y1 = w1(x)
                        y2 = w2(y1)
                        out = w3(y2)
                   @return out
                   end
    return model_stage4
end
function ModelResNet50()
    model_resnet50 = Chain(ModelStage0(),
                           ModelStage1(),
                           ModelStage2(),
                           ModelStage3(),
                           ModelStage4())
    return model_resnet50
end
