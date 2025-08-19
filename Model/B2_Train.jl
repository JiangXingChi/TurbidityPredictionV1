include("A3_BackBone.jl")
include("B1_DataTool.jl")
function OneTrain(; model, ps, st, opt, ad, loss_function, data, dev)
    ps = ps |> dev
    st = st |> dev
    data = data |> dev
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    gs, loss, stats, train_state = Training.single_train_step!(ad, loss_function, data, train_state)
    ps = train_state.parameters
    st = train_state.states
    dev_cpu = cpu_device()
    ps = ps |> dev_cpu
    st = st |> dev_cpu
    loss = loss |> dev_cpu
    return ps, st, loss
end