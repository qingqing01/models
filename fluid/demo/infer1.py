# 1. Network
img = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
pred = lenet_5(img)
test_program = fluid.default_main_program().clone(for_test=True)
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# 2. Load model
fluid.io.load_persistables(exe, 'mnist_model', main_program=test_program)


# 3. Load image
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


im = load_image('test.png')
# 4. Run
results, = exe.run(test_program, feed={'img': im}, fetch_list=[pred])
num = np.argmax(results)
prob = results[0][num]
print("Inference result, prob: {}, number {}".format(prob, num))
