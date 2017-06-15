import os

# keep_probs = [i/10.0 for i in range(1, 11)]
keep_probs = [0.8]
# hidden_sizes = [16, 32, 64, 128, 256]
hidden_sizes = [256, 512, 1024]
lstm_forget_biases = [0.1, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 8.0]
for keep_prob in keep_probs:
    for hidden_size in hidden_sizes:
        for forget_biase in lstm_forget_biases:
            try:
                os.system('python /home/shangmingyang/projects/MVModel/rnn/rnn_model.py --keep_prob=%f --hidden_size=%d --forget_biases=%f --acc_result_file=%s'
                %(keep_prob, hidden_size, forget_biase,'/home/shangmingyang/projects/MVModel/rnn/sigmoid49_lstm_forget_100epoch.csv'))
            except Exception as e:
                print("Exception:", e)

