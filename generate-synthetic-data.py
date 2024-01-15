from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from pia_functions import data_train_test
import pandas as pd
import numpy as np

#X_train, X_test, y_train, y_test, sensitive, sensitive_t = data_train_test()
output_size = 20000
#metadata = SingleTableMetadata()
#to_fit = pd.DataFrame(np.concatenate((X_train, X_test)), columns=[str(i) for i in range(79)])
#metadata.detect_from_dataframe(data=to_fit)

#syn_model = CTGANSynthesizer(metadata)
#syn_model.fit(to_fit)
syn_model = CTGANSynthesizer.load('syn_model')
#syn_model.save('syn_model')

sampled = syn_model.sample(num_rows=output_size)

# save data
sampled.to_csv("data/syn_data.csv", index=False)