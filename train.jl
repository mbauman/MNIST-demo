# Classifies MNIST digits with a convolutional network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exit, and learning rate scheduling.
#
# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

include("model.jl")

args, t = @timed train()
@show t
@time acc = test()
@show acc

ENV["OUTPUTS"] = """{
    "accuracy": $acc,
    "epochs": $(args.epochs),
    "time": $t
}"""

ENV["RESULTS_FILE_TO_UPLOAD"] = joinpath(args.savepath, "mnist_conv.bson")
