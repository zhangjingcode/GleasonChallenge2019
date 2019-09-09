def DownSampling(raw_data, ratio=10):
    pass

def UpSampling(processed_data, ratio, process_log):
    pass

shape = (5120, 5120, 1)
shape = (5120, 4800, 1)
shape = (5121, 5120, 1)
shape = (5121, 5121, 1)

raw_label = np.zeros((5000, 4800, 1))
processed_label, process_log = DownSampling(raw_label, ratio=10)
generate_new_label = UpSampling(processed_label, ratio=10, process_log)

assert(generate_new_label - raw_label == 0)