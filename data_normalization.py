import data_visualization





visualize()

window = 22
X_train, y_train, X_test, y_test = load_data(df, window)
print (X_train[0], y_train[0])
model = build_model([5,window,1])


