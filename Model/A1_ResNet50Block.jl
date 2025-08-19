using Lux
function GetChannel(x::AbstractArray)
    channel = size(x)[3]
    return channel
end
function ModelBtnk1(c::Int, c1::Int, s::Int)
    model_btnk1 = @compact(wl1 = Conv((1,1), c=>c1; stride=s, pad=0),
                           wl2 = BatchNorm(c1, relu),
                           wl3 = Conv((3,3), c1=>c1; stride=1, pad=1),
                           wl4 = BatchNorm(c1, relu),
                           wl5 = Conv((1,1), c1=>c1*4; stride=1, pad=0),
                           wl6 = BatchNorm(c1*4),
                           wr1 = Conv((1,1), c=>c1*4; stride=s, pad=0),
                           wr2 = BatchNorm(c1*4)
                          ) do x
                    yl1 = wl1(x)
                    yl2 = wl2(yl1)
                    yl3 = wl3(yl2)
                    yl4 = wl4(yl3)
                    yl5 = wl5(yl4)
                    yl6 = wl6(yl5)
                    yr1 = wr1(x)
                    yr2 = wr2(yr1)
                    out = relu(yl6 + yr2)
                  @return out
                  end
    return model_btnk1
end
function ModelBtnk2(c::Int)
    model_btnk2 = @compact(w1 = Conv((1,1), c=>Int(c/4); stride=1, pad=0),
                           w2 = BatchNorm(Int(c/4), relu),
                           w3 = Conv((3,3), Int(c/4)=>Int(c/4); stride=1, pad=1),
                           w4 = BatchNorm(Int(c/4), relu),
                           w5 = Conv((1,1), Int(c/4)=>c); stride=1, pad=0,
                           w6 = BatchNorm(c)) do x
                    y1 = w1(x)
                    y2 = w2(y1)
                    y3 = w3(y2)
                    y4 = w4(y3)
                    y5 = w5(y4)
                    y6 = w6(y5)
                    out = relu(y6 + x)
                  @return out
                  end
    return model_btnk2
end