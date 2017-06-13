import os

keep_probs = [i/10.0 for i in range(1, 11)]
hidden_sizes = [16, 32, 64, 128, 256]
for keep_prob in keep_probs:
    for hidden_size in hidden_sizes:
        try:
            os.system('python /home/shangmingyang/projects/TF/rnn/rnn_model_copy.py --keep_prob=%f --hidden_size=%d --acc_result_file=%s' %(keep_prob, hidden_size, '/home/shangmingyang/projects/TF/rnn/sigmoid49_lstm_keep_hidden_100epoch.csv'))
        except e:
            print("Exception:", e)

