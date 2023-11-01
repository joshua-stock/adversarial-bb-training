import pandas as pd
import numpy as np
from pia_functions import get_distributed_adult_sets, generate_shadow_model_outputs, data_train_test

n_shadow_models=200
distributions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
X_train, X_test, y_train, y_test, sensitive, sensitive_t = data_train_test()

distributed_datasets = get_distributed_adult_sets(distributions=distributions)
all_shadow_outputs = []
for ds in distributed_datasets:
    print(f"now generating {ds.distribution}...")
    outputs = generate_shadow_model_outputs(ds, X_test, n_shadow_models=n_shadow_models)
    all_shadow_outputs.append(outputs)

adv_df = pd.DataFrame(np.array(np.concatenate((all_shadow_outputs))))
adv_df["y"] = np.concatenate(([np.repeat(d, n_shadow_models) for d in distributions]))
adv_df.to_csv("shadow_model_outputs.csv")