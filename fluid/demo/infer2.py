def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# 1. Load model
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
model_path = 'mnist_save_model'
infer_program, feeds, fetches = fluid.io.load_inference_model(
    model_path, exe, model_filename='model', params_filename='params')
# 2. Load image
im = load_image('test.png')
# 3. Run
results, = exe.run(infer_program, feed={feeds[0]: im}, fetch_list=fetches)
num = np.argmax(results)
prob = results[0][num]
print("Inference result, prob: {}, number {}".format(prob, num))
