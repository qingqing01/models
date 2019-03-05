log_writter = LogWriter("./vdl_log", sync_cycle=10)
with log_writter.mode("train") as logger:
    trn_scalar_loss = logger.scalar(tag="loss")
    trn_scalar_acc = logger.scalar(tag="acc")
with log_writter.mode('test') as logger:
    tst_scalar_loss = logger.scalar(tag="loss")
    tst_scalar_acc = logger.scalar(tag="acc")

# start to train
for i in range(epochs):
    train_acc, train_cost = [], []
    for step, batch in enumerate(train_reader()):
        res_cost, res_acc = exe.run(fluid.default_main_program(),
                                    feed=feeder.feed(batch),
                                    fetch_list=[avg_cost.name, acc.name])
        train_cost.append(res_cost)
        train_acc.append(res_acc)

        if step % 50 == 0:
            # record the loss and accuracy
            mloss = np.mean(np.array(train_cost))
            macc = np.mean(np.array(train_acc))
            trn_scalar_loss.add_record(step, mloss)
            trn_scalar_acc.add_record(step, macc)
            train_acc, train_cost = [], []
            print("Epoc:{}, Iter:{}, loss:{}, acc{}".format(i, step, mloss, macc))

            test_acc, test_cost = [], []
            for data in test_reader():
                res_cost, res_acc = exe.run(
                    test_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost.name, acc.name])
                test_cost.append(res_cost)
                test_acc.append(res_acc)
                mloss = np.mean(np.array(test_cost))
                macc = np.mean(np.array(test_acc))
                tst_scalar_loss.add_record(step, mloss)
                tst_scalar_acc.add_record(step, macc)
                test_acc, test_cost = [], []
                print("Test Epoc:{}, loss:{}, acc{}".format(i, mloss, macc))
