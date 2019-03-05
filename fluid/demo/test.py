# 1. network
img = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
avg_cost, acc = lenet_5(img, label)
test_program = fluid.default_main_program().clone(for_test=True)

# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
exe = fluid.Executor(place)
# init all param
exe.run(fluid.default_startup_program())

# 2. Load model
fluid.io.load_persistables(exe, 'mnist_model', main_program=test_program)
# 3. Define reader
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)
# 4. Run
test_acc, test_cost = [], []
for data in test_reader():
    res_cost, res_acc = exe.run(test_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost.name, acc.name])
    test_cost.append(res_cost)
    test_acc.append(res_acc)
mloss = np.mean(np.array(test_cost))
macc = np.mean(np.array(test_acc))
print("Test loss:{}, acc: {}".format(mloss, macc))
