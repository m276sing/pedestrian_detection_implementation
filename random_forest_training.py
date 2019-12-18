from sklearn.ensemble import RandomForestClassifier
import pickle

print("no training")
features_of_images = pickle.load(open("/Users/15195/Desktop/ece613/ped_features.p", 'rb'))
labels_of_images = pickle.load(open("/Users/15195/Desktop/ece613/peds_feature_to_label.p", 'rb'))
print("half training")

random_forest = RandomForestClassifier(1000)
random_forest_model = random_forest.fit(features_of_images, labels_of_images)

print("pending training")
pickle.dump(random_forest_model, open("/Users/15195/Desktop/ece613/Random_forest_trained_model.p", 'wb'))
print("done training")
