#grid search - use for tuning models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
epochs = 100

def create_model(nodes, noise_amount):

  model_noise_hid = Sequential()
  model_noise_hid.add(GRU(nodes, return_sequences=True, input_shape=(tim_steps, n_feats), recurrent_dropout=0.05))
  model_noise_hid.add(GaussianNoise(noise_amount))
  model_noise_hid.add(GRU(6, return_sequences=False, recurrent_dropout=0.05))
  model_noise_hid.add(GaussianNoise(0.01))
  model_noise_hid.add(Dense(1, activation='sigmoid'))
  model_noise_hid.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model_noise_hid

grid_values = {'noise_amount':[0.001, 0.01],
               'nodes': [1, 16]}

model = KerasClassifier(build_fn=create_model, epochs=epochs, verbose=1, )

grid_model = GridSearchCV(model, param_grid = grid_values)
grid_model.fit(X_train, y_train)

print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
means = grid_model.cv_results_['mean_test_score']
stds = grid_model.cv_results_['std_test_score']
params = grid_model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))