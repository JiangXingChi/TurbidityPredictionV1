# 1 Basic settings
# Get the directory where the script is located
script_dir = dirname(@__FILE__)
# Activate the environment
using Pkg; Pkg.activate(joinpath(script_dir, "Env"))
# Load packages
using JLD2
using Random
# Load the script
include("Model/B2_Train.jl")
# Define the path to the best model
best_model_path = joinpath(script_dir, "Result/best_model.jld2")
# Define the test folder
test_folder = joinpath(script_dir, "Data")

# 2 Using the model
# Load the model
model = ModelBackbone(sample_length=24)
# If you need to load training data, please read the readme and download the parameters
#@load best_model_path ps st
# Load random data
ps,st = Lux.setup(Random.default_rng(0), model)
# Perform validation using a loop
data = BatchData(train_folder=test_folder, id_array=[1])
# Extract actual turbidity values based on image names
y_real = data[2]
# Predict turbidity
y_pred = model(data[1],ps,Lux.testmode(st))[1]